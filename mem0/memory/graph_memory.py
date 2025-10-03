import logging
from datetime import datetime
import pytz
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

from mem0.memory.utils import format_entities

try:
    from langchain_neo4j import Neo4jGraph
except ImportError:
    raise ImportError(
        "langchain_neo4j is not installed. Please install it using pip install langchain-neo4j"
    )

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError(
        "rank_bm25 is not installed. Please install it using pip install rank-bm25"
    )

from mem0.graphs.tools import (
    ANALYZE_RELATION_EVOLUTION_TOOL,
    DELETE_MEMORY_STRUCT_TOOL_GRAPH,
    DELETE_MEMORY_TOOL_GRAPH,
    EXTRACT_ENTITIES_STRUCT_TOOL,
    EXTRACT_ENTITIES_TOOL,
    RELATIONS_STRUCT_TOOL,
    RELATIONS_TOOL,
)
from mem0.graphs.utils import EXTRACT_RELATIONS_PROMPT, get_delete_messages
from mem0.utils.factory import EmbedderFactory, LlmFactory

logger = logging.getLogger(__name__)


class MemoryGraph:
    def __init__(self, config):
        self.config = config
        self.graph = Neo4jGraph(
            self.config.graph_store.config.url,
            self.config.graph_store.config.username,
            self.config.graph_store.config.password,
        )
        self.embedding_model = EmbedderFactory.create(
            self.config.embedder.provider, self.config.embedder.config
        )

        self.llm_provider = "openai_structured"
        if self.config.llm.provider:
            self.llm_provider = self.config.llm.provider
        if self.config.graph_store.llm:
            self.llm_provider = self.config.graph_store.llm.provider

        self.llm = LlmFactory.create(self.llm_provider, self.config.llm.config)
        self.user_id = None
        self.threshold = 0.7
        
        # Thread pool for background weight adjustments
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="weight_adjuster")

    def _analyze_relation_evolution(self, current_relation, session_history, graph_context, user_id):
        """
        Analyzes the evolution of a single relationship.
        """
        tool_input = {
            "entity": current_relation["source"], # Or destination, depending on context
            "current_relation": {
                "relationship": current_relation["relatationship"],
                "emotion": current_relation.get("emotion", "neutral"),
                "weight": current_relation.get("weight", "relevant"),
                "status": current_relation.get("status", "active"),
                "last_mentioned": current_relation.get("last_mentioned", ""),
                "usage_count": current_relation.get("usage_count", 1)
            },
            "session_history": [session_history],
            "graph_context": graph_context
        }

        messages = [
            {
                "role": "system", 
                "content": f"""You are an expert in analyzing user relationships with entities. Your task is to infer behavioral and emotional changes based on recent conversation history and graph context.

Analyze the relationship evolution for user {user_id}:

1. **Analyze Emotional Tone**: Compare the user's current language to their previous emotional state with this entity.
2. **Analyze Importance**: Determine if this relationship has become more or less important/significant to the user.
3. **Analyze Context**: Consider if other relationships are displacing this one or if this is becoming an obsession.
4. **Weight Categories**: Weight represents how much this relationship matters to the user:
   - "ignored": Completely unimportant, user dismisses or avoids
   - "peripheral": Minor importance, rarely mentioned
   - "transitional": Temporary or changing importance
   - "relevant": Standard importance, regularly mentioned
   - "ritualistic": Part of routine or habit
   - "important": High significance in user's life
   - "core_identity": Central to who the user is
   - "infatuation": Intense but potentially temporary fascination
   - "devotion": Deep, committed attachment
   - "obsession": Overwhelming preoccupation
   - "repressed": Suppressed but significant relationship
   - "negative_core": Important but negative relationship

Return updated weight, emotion, status, and analysis flags."""
            },
            {
                "role": "user", 
                "content": f"Analyze the relationship for the entity '{tool_input['entity']}'. Current relationship data: {tool_input['current_relation']}. Recent user messages: {tool_input['session_history']}. Broader context: {tool_input['graph_context']}"
            }
        ]
        
        _tools = [ANALYZE_RELATION_EVOLUTION_TOOL]

        response = self.llm.generate_response(
            messages=messages,
            tools=_tools,
        )

        if response and response.get("tool_calls"):
            tool_call = response["tool_calls"][0]
            if tool_call["name"] == "analyze_relation_evolution":
                return tool_call["arguments"]
        
        return None

    def analyze_and_update_existing_relations(self, search_output, data, filters):
        """
        Analyzes and updates existing relationships based on new data.
        """
        updated_relations = []
        if not search_output:
            return updated_relations

        # Create graph_context from all relationships found
        graph_context = [f"{r['source']}-{r['relatationship']}->{r['destination']}" for r in search_output]

        for i, relation in enumerate(search_output):
            evolution_result = self._analyze_relation_evolution(relation, data, graph_context, filters["user_id"])
            
            if not evolution_result:
                continue

            # Validate that all required keys are present
            required_keys = ["weight", "emotion", "status", "has_emotional_shift", "has_habit_changed"]
            if not all(key in evolution_result for key in required_keys):
                logger.warning(f"LLM analysis result for {relation['source']} is missing required keys. Skipping evolution update. Result: {evolution_result}")
                continue

            if evolution_result["has_emotional_shift"] or evolution_result["has_habit_changed"]:
                logger.info(f"Detected evolution in relationship: {relation['source']} -> {relation['relatationship']} -> {relation['destination']}")
                
                # Prepare properties for update
                update_payload = {
                    "weight": evolution_result["weight"],
                    "emotion": evolution_result["emotion"],
                    "status": evolution_result["status"],
                }
                
                # Call update_relationship
                updated_rel = self.update_relationship(
                    relation['source'], 
                    relation['relatationship'], 
                    relation['destination'], 
                    filters["user_id"], 
                    **update_payload
                )
                
                if updated_rel:
                    updated_relations.append(updated_rel)
        
        return updated_relations

    def add(self, data, filters):
        """
        Adds data to the graph.

        Args:
            data (str): The data to add to the graph.
            filters (dict): A dictionary containing filters to be applied during the addition.
        """
        # Step 1: Retrieve nodes from data
        entity_type_map = self._retrieve_nodes_from_data(data, filters)
        
        # Step 2: Establish relations from data
        to_be_added = self._establish_nodes_relations_from_data(
            data, filters, entity_type_map
        )
        
        # Step 3: Search graph database
        search_output = self._search_graph_db(
            node_list=list(entity_type_map.keys()), filters=filters
        )
        
        # Step 4: Analyze and update existing relations
        # NOTE: Weight adjustment is now done during search operations, not during add
        # This makes add operations faster and analyzes weights based on actual usage
        # evolution_updates = self.analyze_and_update_existing_relations(search_output, data, filters)
        evolution_updates = []  # Disabled - now handled in search
        
        # Step 5: Get delete entities from search output
        to_be_updated = self._get_delete_entities_from_search_output(
            search_output, data, filters
        )

        # TODO: Batch queries with APOC plugin
        # TODO: Add more filter support
        
        # Step 6: Process relationship updates
        updated_entities = self._process_relationship_updates(to_be_updated, filters["user_id"])
        
        updated_entities.extend(evolution_updates)
        
        # Step 7: Add entities
        added_entities = self._add_entities(
            to_be_added, filters["user_id"], entity_type_map
        )

        return {"updated_entities": updated_entities, "added_entities": added_entities}

    def _background_weight_adjustment(self, search_results, query, filters):
        """
        Background task to analyze and update weights of relationships based on search results.
        This runs asynchronously and doesn't block the search response.
        """
        try:
            logger.info(f"Starting background weight adjustment for {len(search_results)} search results")
            
            # Convert search results back to the format expected by analyze_and_update_existing_relations
            # Note: search results have 'relationship' while internal format uses 'relatationship'
            search_output_format = []
            for result in search_results:
                formatted_result = {
                    "source": result["source"],
                    "relatationship": result["relationship"],  # Map back to internal format
                    "destination": result.get("destination") or result.get("target"),
                }
                
                # Copy all other properties
                for key in ["weight", "is_uncertain", "status", "start_date", "end_date", 
                           "emotion", "last_mentioned", "usage_count"]:
                    if key in result:
                        formatted_result[key] = result[key]
                
                search_output_format.append(formatted_result)
            
            # Use the search query as the session history/context for weight analysis
            # This gives the LLM context about what the user was looking for
            updated_relations = self.analyze_and_update_existing_relations(
                search_output_format, 
                query,  # Use search query as the "data" parameter
                filters
            )
            
            logger.info(f"Background weight adjustment completed. Updated {len(updated_relations)} relationships")
            
        except Exception as e:
            logger.error(f"Error in background weight adjustment: {e}", exc_info=True)

    def search(self, query, filters, limit=100):
        """
        Search for memories and related graph data.

        Args:
            query (str): Query to search for.
            filters (dict): A dictionary containing filters to be applied during the search.
            limit (int): The maximum number of nodes and relationships to retrieve. Defaults to 100.

        Returns:
            dict: A dictionary containing:
                - "contexts": List of search results from the base data store.
                - "entities": List of related graph data based on the query.
        """
        # Step 1: Retrieve nodes from query
        entity_type_map = self._retrieve_nodes_from_data(query, filters)
        
        # Step 2: Search graph database
        search_output = self._search_graph_db(
            node_list=list(entity_type_map.keys()), filters=filters
        )

        if not search_output:
            return []

        # Step 3: Prepare for BM25 ranking
        search_outputs_sequence = [
            [item["source"], item["relatationship"], item["destination"]]
            for item in search_output
        ]
        bm25 = BM25Okapi(search_outputs_sequence)

        tokenized_query = query.split(" ")
        reranked_results = bm25.get_top_n(tokenized_query, search_outputs_sequence, n=15)

        # Step 4: Build final results and update metadata
        search_results = []
        current_time_iso = datetime.now(pytz.utc).isoformat()
        
        for item in reranked_results:
            # Find the original item to retrieve all properties
            for orig_item in search_output:
                if (
                    orig_item["source"] == item[0]
                    and orig_item["relatationship"] == item[1]
                    and orig_item["destination"] == item[2]
                ):

                    result_dict = {
                        "source": item[0],
                        "relationship": item[1],
                        "destination": item[2],
                    }

                    # Add optional parameters if they exist
                    if orig_item.get("weight") is not None:
                        result_dict["weight"] = orig_item["weight"]
                    if orig_item.get("is_uncertain") is not None:
                        result_dict["is_uncertain"] = orig_item["is_uncertain"]
                    if orig_item.get("status") is not None:
                        result_dict["status"] = orig_item["status"]
                    if orig_item.get("start_date") is not None:
                        result_dict["start_date"] = orig_item["start_date"]
                    if orig_item.get("end_date") is not None:
                        result_dict["end_date"] = orig_item["end_date"]
                    if orig_item.get("emotion") is not None:
                        result_dict["emotion"] = orig_item["emotion"]
                    if orig_item.get("last_mentioned") is not None:
                        result_dict["last_mentioned"] = orig_item["last_mentioned"]
                    if orig_item.get("usage_count") is not None:
                        result_dict["usage_count"] = orig_item["usage_count"]

                    # Update mention metadata for this selected relationship
                    self.update_mention_metadata(result_dict, current_time_iso, filters["user_id"])
                    
                    # Update mention metadata for nodes referenced in this relationship
                    self.update_node_mention_metadata(item[0], current_time_iso, filters["user_id"])
                    self.update_node_mention_metadata(item[2], current_time_iso, filters["user_id"])

                    search_results.append(result_dict)
                    break
            else:
                # Fallback if original item not found
                fallback_result = {"source": item[0], "relationship": item[1], "destination": item[2]}
                # Still update metadata even for fallback case
                self.update_mention_metadata(fallback_result, current_time_iso, filters["user_id"])
                self.update_node_mention_metadata(item[0], current_time_iso, filters["user_id"])
                self.update_node_mention_metadata(item[2], current_time_iso, filters["user_id"])
                search_results.append(fallback_result)

        logger.info(f"Returned {len(search_results)} search results")

        # Trigger weight adjustment in the background without blocking
        # Pass the search query as context for better weight analysis
        self._executor.submit(
            self._background_weight_adjustment, 
            search_results.copy(),  # Copy to avoid modification issues
            query,  # Pass the search query as context
            filters
        )

        return search_results

    def delete_all(self, filters):
        cypher = """
        MATCH (n {user_id: $user_id})
        DETACH DELETE n
        """
        params = {"user_id": filters["user_id"]}
        self.graph.query(cypher, params=params)

    def get_all(self, filters, limit=100):
        """
        Retrieves all nodes and relationships from the graph database based on optional filtering criteria.

        Args:
            filters (dict): A dictionary containing filters to be applied during the retrieval.
            limit (int): The maximum number of nodes and relationships to retrieve. Defaults to 100.
        Returns:
            list: A list of dictionaries, each containing:
                - 'contexts': The base data store response for each memory.
                - 'entities': A list of strings representing the nodes and relationships
        """
        # return all nodes and relationships
        query = """
        MATCH (n {user_id: $user_id})-[r]->(m {user_id: $user_id})
        RETURN 
            n.name AS source, 
            type(r) AS relationship, 
            m.name AS target,
            r.weight AS weight,
            r.is_uncertain AS is_uncertain,
            r.status AS status,
            r.start_date AS start_date,
            r.end_date AS end_date,
            r.emotion AS emotion,
            r.last_mentioned AS last_mentioned,
            r.usage_count AS usage_count
        LIMIT $limit
        """
        
        results = self.graph.query(
            query, params={"user_id": filters["user_id"], "limit": limit}
        )

        final_results = []
        current_time_iso = datetime.now(pytz.utc).isoformat()
        
        for result in results:
            result_dict = {
                "source": result["source"],
                "relationship": result["relationship"],
                "target": result["target"],
            }

            # Add optional parameters if they exist in the result
            if result.get("weight") is not None:
                result_dict["weight"] = result["weight"]
            if result.get("is_uncertain") is not None:
                result_dict["is_uncertain"] = result["is_uncertain"]
            if result.get("status") is not None:
                result_dict["status"] = result["status"]
            if result.get("start_date") is not None:
                result_dict["start_date"] = result["start_date"]
            if result.get("end_date") is not None:
                result_dict["end_date"] = result["end_date"]
            if result.get("emotion") is not None:
                result_dict["emotion"] = result["emotion"]
            if result.get("last_mentioned") is not None:
                result_dict["last_mentioned"] = result["last_mentioned"]
            if result.get("usage_count") is not None:
                result_dict["usage_count"] = result["usage_count"]

            # Update mention metadata for this retrieved relationship
            self.update_mention_metadata(result_dict, current_time_iso, filters["user_id"])
            
            # Update mention metadata for nodes referenced in this relationship
            self.update_node_mention_metadata(result["source"], current_time_iso, filters["user_id"])
            self.update_node_mention_metadata(result["target"], current_time_iso, filters["user_id"])

            final_results.append(result_dict)

        logger.info(f"Retrieved {len(final_results)} relationships")

        return final_results

    def _retrieve_nodes_from_data(self, data, filters):
        """Extracts all the entities mentioned in the query."""
        _tools = [EXTRACT_ENTITIES_TOOL]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [EXTRACT_ENTITIES_STRUCT_TOOL]
        
        search_results = self.llm.generate_response(
            messages=[
                {
                    "role": "system",
                    "content": f"You are a smart assistant who understands entities and their types in a given text. If user message contains self reference such as 'I', 'me', 'my' etc. then use {filters['user_id']} as the source entity. Extract all the entities from the text. ***DO NOT*** answer the question itself if the given text is a question.",
                },
                {"role": "user", "content": data},
            ],
            tools=_tools,
        )

        entity_type_map = {}

        try:
            for item in search_results["tool_calls"][0]["arguments"]["entities"]:
                entity_type_map[item["entity"]] = item["entity_type"]
        except Exception as e:
            logger.exception(
                f"Error in search tool: {e}, llm_provider={self.llm_provider}, search_results={search_results}"
            )

        entity_type_map = {
            k.lower().replace(" ", "_"): v.lower().replace(" ", "_")
            for k, v in entity_type_map.items()
        }
        
        logger.debug(f"Entity type map: {entity_type_map}, search_results={search_results}")
        return entity_type_map

    def _establish_nodes_relations_from_data(self, data, filters, entity_type_map):
        """Eshtablish relations among the extracted nodes."""
        if self.config.graph_store.custom_prompt:
            messages = [
                {
                    "role": "system",
                    "content": EXTRACT_RELATIONS_PROMPT.replace(
                        "USER_ID", filters["user_id"]
                    ).replace(
                        "CUSTOM_PROMPT", f"4. {self.config.graph_store.custom_prompt}"
                    ),
                },
                {"role": "user", "content": data},
            ]
        else:
            messages = [
                {
                    "role": "system",
                    "content": EXTRACT_RELATIONS_PROMPT.replace(
                        "USER_ID", filters["user_id"]
                    ),
                },
                {
                    "role": "user",
                    "content": f"List of entities: {list(entity_type_map.keys())}. \n\nText: {data}",
                },
            ]

        _tools = [RELATIONS_TOOL]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [RELATIONS_STRUCT_TOOL]

        extracted_entities = self.llm.generate_response(
            messages=messages,
            tools=_tools,
        )

        if extracted_entities["tool_calls"]:
            extracted_entities = extracted_entities["tool_calls"][0]["arguments"][
                "entities"
            ]
            # Log emotions for debugging - using mem0 format
            for entity in extracted_entities:
                emotion = entity.get("emotion")
                logger.info(f"LLM extracted: {entity.get('source', '?')} -> {entity.get('relationship', '?')} -> {entity.get('destination', '?')}, emotion='{emotion}'")
        else:
            extracted_entities = []

        extracted_entities = self._remove_spaces_from_entities(extracted_entities)
        
        logger.debug(f"Extracted entities: {extracted_entities}")
        return extracted_entities

    def _search_graph_db(self, node_list, filters, limit=100):
        """Search similar nodes among and their respective incoming and outgoing relations."""
        if not node_list:
            return []

        # Prepare a list of nodes with their embeddings to pass as a single parameter
        nodes_with_embeddings = [
            {"name": node, "embedding": self.embedding_model.embed(node)}
            for node in node_list
        ]

        # This single, optimized query implements the "Anchor-Expand" pattern.
        # EFFICIENT: Similarity calculation happens only once in Step 2 to find anchors.
        # The UNION in Step 3 does NOT recalculate similarity - it just expands from pre-found anchors.
        cypher_query = """
        // Step 1: UNWIND the list of nodes to process them in a batch.
        UNWIND $nodes_with_embeddings AS search_item
        
        // Step 2: Find "anchor" nodes via vector similarity search (ONCE ONLY).
        MATCH (n)
        WHERE n.embedding IS NOT NULL AND n.user_id = $user_id
        WITH search_item, n, round(2 * vector.similarity.cosine(n.embedding, search_item.embedding) - 1, 4) AS similarity
        WHERE similarity >= $threshold

        // Order by similarity and limit to the top N results for each search item
        WITH search_item, n, similarity
        ORDER BY similarity DESC
        WITH search_item, collect({n: n, similarity: similarity})[..$limit] AS top_nodes
        UNWIND top_nodes AS top_node
        WITH top_node.n AS n, top_node.similarity AS similarity

        // Step 3: From the anchors, expand bidirectionally using a CALL subquery.
        // EFFICIENT: No redundant similarity calculation here - 'n' is already the anchor.
        CALL (n) {
            MATCH (n)-[r]->(m)
            WHERE m.user_id = $user_id
            RETURN n.name AS source, elementId(n) AS source_id, type(r) AS relatationship,
                   elementId(r) AS relation_id, m.name AS destination, elementId(m) AS destination_id,
                   r.weight AS weight, r.is_uncertain AS is_uncertain, r.status AS status,
                   r.start_date AS start_date, r.end_date AS end_date, r.emotion AS emotion,
                   r.last_mentioned AS last_mentioned, r.usage_count AS usage_count
            UNION
            MATCH (m)-[r]->(n)
            WHERE m.user_id = $user_id
            RETURN m.name AS source, elementId(m) AS source_id, type(r) AS relatationship,
                   elementId(r) AS relation_id, n.name AS destination, elementId(n) AS destination_id,
                   r.weight AS weight, r.is_uncertain AS is_uncertain, r.status AS status,
                   r.start_date AS start_date, r.end_date AS end_date, r.emotion AS emotion,
                   r.last_mentioned AS last_mentioned, r.usage_count AS usage_count
        }
        // Step 4: De-duplicate and return the final results with all rich metadata.
        WITH DISTINCT source, source_id, relatationship, relation_id, destination, destination_id, similarity,
             weight, is_uncertain, status, start_date, end_date, emotion, last_mentioned, usage_count
        RETURN source, source_id, relatationship, relation_id, destination, destination_id, similarity,
               weight, is_uncertain, status, start_date, end_date, emotion, last_mentioned, usage_count
        """

        params = {
            "nodes_with_embeddings": nodes_with_embeddings,
            "threshold": self.threshold,
            "user_id": filters["user_id"],
            "limit": limit,
        }

        result_relations = self.graph.query(cypher_query, params=params)

        return result_relations

    def _get_delete_entities_from_search_output(self, search_output, data, filters):
        """Get the entities to be deleted from the search output."""
        search_output_string = format_entities(search_output)
        system_prompt, user_prompt = get_delete_messages(
            search_output_string, data, filters["user_id"]
        )

        _tools = [DELETE_MEMORY_TOOL_GRAPH]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [
                DELETE_MEMORY_STRUCT_TOOL_GRAPH,
            ]

        memory_updates = self.llm.generate_response(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tools=_tools,
        )
        
        to_be_updated = []
        for item in memory_updates["tool_calls"]:
            if item["name"] == "delete_graph_memory":
                to_be_updated.append(item["arguments"])
        # in case if it is not in the correct format
        to_be_updated = self._remove_spaces_from_entities(to_be_updated)
        
        logger.debug(f"Deleted relationships: {to_be_updated}")
        return to_be_updated

    def _process_relationship_updates(self, to_be_updated, user_id):
        """Update the status of relationships in the graph, marking them as ended or invalid."""
        results = []
        for i, item in enumerate(to_be_updated):
            source = item["source"]
            destination = item["destination"]
            relationship_type = item["relationship"]
            status = item.get("status", "ended")  # Default to 'ended'

            properties_to_update = {"status": status}
            
            if status in ["ended", "invalid"]:
                properties_to_update["end_date"] = datetime.now(pytz.utc).isoformat()
            
            updated_rel = self.update_relationship(
                source, relationship_type, destination, user_id, **properties_to_update
            )
            
            if updated_rel and updated_rel.get("status") != "not_found":
                results.append(updated_rel)
            else:
                logger.debug(
                    f"Relationship {relationship_type} between {source} and {destination} not found for update."
                )
                results.append(
                    {
                        "source": source,
                        "relationship": relationship_type,
                        "target": destination,
                        "status": "not_found",
                    }
                )
        
        logger.debug(f"Updated relationships: {results}")
        return results

    def _add_entities(self, to_be_added, user_id, entity_type_map):
        """Add the new entities to the graph. Merge the nodes if they already exist."""
        results = []
        logger.debug(f"Adding entities. `to_be_added`: {to_be_added}")
        for i, item in enumerate(to_be_added):
            # entities
            source = item["source"]
            destination = item["destination"]
            relationship = item["relationship"]

            # types
            source_type = entity_type_map.get(source, "unknown")
            destination_type = entity_type_map.get(destination, "unknown")

            # embeddings
            source_embedding = self.embedding_model.embed(source)
            dest_embedding = self.embedding_model.embed(destination)

            # additional parameters
            weight = item.get("weight")
            is_uncertain = item.get("is_uncertain")
            status = item.get("status")
            start_date = item.get("start_date")
            end_date = item.get("end_date")
            emotion = item.get("emotion")
            last_mentioned = item.get("last_mentioned")
            usage_count = item.get("usage_count")

            # search for the nodes with the closest embeddings
            source_node_search_result = self._search_source_node(
                source_embedding, user_id, threshold=0.9
            )
            destination_node_search_result = self._search_destination_node(
                dest_embedding, user_id, threshold=0.9
            )

            logger.debug(f"Processing item: {item}. Source found: {bool(source_node_search_result)}. Destination found: {bool(destination_node_search_result)}")

            # TODO: Create a cypher query and common params for all the cases
            if not destination_node_search_result and source_node_search_result:
                current_formatted_time = datetime.now(pytz.utc).strftime('%Y-%m-%d %H:%M:%S')
                
                # Build SET clauses for relationship properties
                relationship_set_clauses = []
                if weight is not None:
                    relationship_set_clauses.append("r.weight = $weight")
                if is_uncertain is not None:
                    relationship_set_clauses.append("r.is_uncertain = $is_uncertain")
                if status is not None:
                    relationship_set_clauses.append("r.status = $status")
                if start_date is not None:
                    relationship_set_clauses.append("r.start_date = $start_date")
                if end_date is not None:
                    relationship_set_clauses.append("r.end_date = $end_date")
                if emotion is not None:
                    relationship_set_clauses.append("r.emotion = $emotion")
                if last_mentioned is not None:
                    relationship_set_clauses.append("r.last_mentioned = $last_mentioned")
                if usage_count is not None:
                    relationship_set_clauses.append("r.usage_count = $usage_count")
                
                additional_set_properties_str = ""
                if relationship_set_clauses:
                    additional_set_properties_str = ", " + ", ".join(relationship_set_clauses)

                cypher = f"""
                    MATCH (source)
                    WHERE elementId(source) = $source_id
                    MERGE (destination:{destination_type} {{name: $destination_name, user_id: $user_id}})
                    ON CREATE SET
                        destination.created_at = $current_formatted_time,
                        destination.embedding = $destination_embedding
                    ON MATCH SET
                        destination.embedding = $destination_embedding
                    MERGE (source)-[r:{relationship}]->(destination)
                    ON CREATE SET 
                        r.created_at = $current_formatted_time,
                        r.usage_count = 1,
                        r.last_mentioned = $current_formatted_time{additional_set_properties_str}
                    ON MATCH SET
                        r.last_mentioned = $current_formatted_time,
                        r.usage_count = COALESCE(r.usage_count, 0) + 1{additional_set_properties_str}
                    RETURN source.name AS source, type(r) AS relationship, destination.name AS target
                    """

                params = {
                    "source_id": source_node_search_result[0][
                        "elementId(source_candidate)"
                    ],
                    "destination_name": destination,
                    "relationship": relationship,
                    "destination_type": destination_type,
                    "destination_embedding": dest_embedding,
                    "user_id": user_id,
                    "current_formatted_time": current_formatted_time,
                }

                # Add additional parameters to params if they exist
                if weight is not None:
                    params["weight"] = weight
                if is_uncertain is not None:
                    params["is_uncertain"] = is_uncertain
                if status is not None:
                    params["status"] = status
                if start_date is not None:
                    params["start_date"] = start_date
                if end_date is not None:
                    params["end_date"] = end_date
                if emotion is not None:
                    params["emotion"] = emotion
                if last_mentioned is not None:
                    params["last_mentioned"] = last_mentioned
                if usage_count is not None:
                    params["usage_count"] = usage_count

                logger.debug(f"Executing Cypher (source exists): {cypher} with params: {params}")
                resp = self.graph.query(cypher, params=params)
                logger.debug(f"Graph query response: {resp}")
                results.append(resp)

            elif destination_node_search_result and not source_node_search_result:
                current_formatted_time = datetime.now(pytz.utc).strftime('%Y-%m-%d %H:%M:%S')

                # Build SET clauses for relationship properties
                relationship_set_clauses = []
                if weight is not None:
                    relationship_set_clauses.append("r.weight = $weight")
                if is_uncertain is not None:
                    relationship_set_clauses.append("r.is_uncertain = $is_uncertain")
                if status is not None:
                    relationship_set_clauses.append("r.status = $status")
                if start_date is not None:
                    relationship_set_clauses.append("r.start_date = $start_date")
                if end_date is not None:
                    relationship_set_clauses.append("r.end_date = $end_date")
                if emotion is not None:
                    relationship_set_clauses.append("r.emotion = $emotion")
                if last_mentioned is not None:
                    relationship_set_clauses.append("r.last_mentioned = $last_mentioned")
                if usage_count is not None:
                    relationship_set_clauses.append("r.usage_count = $usage_count")

                additional_set_properties_str = ""
                if relationship_set_clauses:
                    additional_set_properties_str = ", " + ", ".join(relationship_set_clauses)

                cypher = f"""
                    MATCH (destination)
                    WHERE elementId(destination) = $destination_id
                    MERGE (source:{source_type} {{name: $source_name, user_id: $user_id}})
                    ON CREATE SET
                        source.created_at = $current_formatted_time,
                        source.embedding = $source_embedding
                    ON MATCH SET
                        source.embedding = $source_embedding
                    MERGE (source)-[r:{relationship}]->(destination)
                    ON CREATE SET 
                        r.created_at = $current_formatted_time,
                        r.usage_count = 1,
                        r.last_mentioned = $current_formatted_time{additional_set_properties_str}
                    ON MATCH SET
                        r.last_mentioned = $current_formatted_time,
                        r.usage_count = COALESCE(r.usage_count, 0) + 1{additional_set_properties_str}
                    RETURN source.name AS source, type(r) AS relationship, destination.name AS target
                    """

                params = {
                    "destination_id": destination_node_search_result[0][
                        "elementId(destination_candidate)"
                    ],
                    "source_name": source,
                    "relationship": relationship,
                    "source_type": source_type,
                    "source_embedding": source_embedding,
                    "user_id": user_id,
                    "current_formatted_time": current_formatted_time,
                }

                # Add additional parameters to params if they exist
                if weight is not None:
                    params["weight"] = weight
                if is_uncertain is not None:
                    params["is_uncertain"] = is_uncertain
                if status is not None:
                    params["status"] = status
                if start_date is not None:
                    params["start_date"] = start_date
                if end_date is not None:
                    params["end_date"] = end_date
                if emotion is not None:
                    params["emotion"] = emotion
                if last_mentioned is not None:
                    params["last_mentioned"] = last_mentioned
                if usage_count is not None:
                    params["usage_count"] = usage_count

                logger.debug(f"Executing Cypher (destination exists): {cypher} with params: {params}")
                resp = self.graph.query(cypher, params=params)
                logger.debug(f"Graph query response: {resp}")
                results.append(resp)

            elif source_node_search_result and destination_node_search_result:
                current_formatted_time = datetime.now(pytz.utc).strftime('%Y-%m-%d %H:%M:%S')

                # Build SET clauses for relationship properties
                relationship_set_clauses = []
                if weight is not None:
                    relationship_set_clauses.append("r.weight = $weight")
                if is_uncertain is not None:
                    relationship_set_clauses.append("r.is_uncertain = $is_uncertain")
                if status is not None:
                    relationship_set_clauses.append("r.status = $status")
                if start_date is not None:
                    relationship_set_clauses.append("r.start_date = $start_date")
                if end_date is not None:
                    relationship_set_clauses.append("r.end_date = $end_date")
                if emotion is not None:
                    relationship_set_clauses.append("r.emotion = $emotion")
                if last_mentioned is not None:
                    relationship_set_clauses.append("r.last_mentioned = $last_mentioned")
                if usage_count is not None:
                    relationship_set_clauses.append("r.usage_count = $usage_count")

                additional_set_properties_str = ""
                # For this case, r.updated_at is also set, so check if relationship_set_clauses is non-empty
                # to decide if a comma is needed before r.updated_at or before the additional properties.
                # However, the original code sets r.created_at and r.updated_at unconditionally on merge.
                # We'll stick to adding optional properties after these.
                if relationship_set_clauses:
                    additional_set_properties_str = ", " + ", ".join(relationship_set_clauses)

                cypher = f"""
                    MATCH (source)
                    WHERE elementId(source) = $source_id
                    MATCH (destination)
                    WHERE elementId(destination) = $destination_id
                    MERGE (source)-[r:{relationship}]->(destination)
                    ON CREATE SET 
                        r.created_at = $current_formatted_time,
                        r.updated_at = $current_formatted_time,
                        r.usage_count = 1,
                        r.last_mentioned = $current_formatted_time{additional_set_properties_str}
                    ON MATCH SET
                        r.updated_at = $current_formatted_time,
                        r.last_mentioned = $current_formatted_time,
                        r.usage_count = COALESCE(r.usage_count, 0) + 1{additional_set_properties_str}
                    RETURN source.name AS source, type(r) AS relationship, destination.name AS target
                    """
                params = {
                    "source_id": source_node_search_result[0][
                        "elementId(source_candidate)"
                    ],
                    "destination_id": destination_node_search_result[0][
                        "elementId(destination_candidate)"
                    ],
                    "user_id": user_id,
                    "relationship": relationship,
                    "current_formatted_time": current_formatted_time,
                }

                # Add additional parameters to params if they exist
                if weight is not None:
                    params["weight"] = weight
                if is_uncertain is not None:
                    params["is_uncertain"] = is_uncertain
                if status is not None:
                    params["status"] = status
                if start_date is not None:
                    params["start_date"] = start_date
                if end_date is not None:
                    params["end_date"] = end_date
                if emotion is not None:
                    params["emotion"] = emotion
                if last_mentioned is not None:
                    params["last_mentioned"] = last_mentioned
                if usage_count is not None:
                    params["usage_count"] = usage_count

                logger.debug(f"Executing Cypher (both exist): {cypher} with params: {params}")
                resp = self.graph.query(cypher, params=params)
                logger.debug(f"Graph query response: {resp}")
                results.append(resp)

            elif not source_node_search_result and not destination_node_search_result:
                current_formatted_time = datetime.now(pytz.utc).strftime('%Y-%m-%d %H:%M:%S')

                # Build SET clauses for relationship properties
                relationship_set_clauses = [] # Note: using 'rel' here as per original query
                if weight is not None:
                    relationship_set_clauses.append("rel.weight = $weight")
                if is_uncertain is not None:
                    relationship_set_clauses.append("rel.is_uncertain = $is_uncertain")
                if status is not None:
                    relationship_set_clauses.append("rel.status = $status")
                if start_date is not None:
                    relationship_set_clauses.append("rel.start_date = $start_date")
                if end_date is not None:
                    relationship_set_clauses.append("rel.end_date = $end_date")
                if emotion is not None:
                    relationship_set_clauses.append("rel.emotion = $emotion")
                if last_mentioned is not None:
                    relationship_set_clauses.append("rel.last_mentioned = $last_mentioned")
                if usage_count is not None:
                    relationship_set_clauses.append("rel.usage_count = $usage_count")
                
                additional_set_properties_str = ""
                if relationship_set_clauses:
                    additional_set_properties_str = ", " + ", ".join(relationship_set_clauses)

                cypher = f"""
                    MERGE (n:{source_type} {{name: $source_name, user_id: $user_id}})
                    ON CREATE SET 
                        n.created_at = $current_formatted_time, 
                        n.embedding = $source_embedding
                    ON MATCH SET 
                        n.embedding = $source_embedding
                    MERGE (m:{destination_type} {{name: $dest_name, user_id: $user_id}})
                    ON CREATE SET 
                        m.created_at = $current_formatted_time, 
                        m.embedding = $dest_embedding
                    ON MATCH SET 
                        m.embedding = $dest_embedding
                    MERGE (n)-[rel:{relationship}]->(m)
                    ON CREATE SET 
                        rel.created_at = $current_formatted_time,
                        rel.usage_count = 1,
                        rel.last_mentioned = $current_formatted_time{additional_set_properties_str}
                    ON MATCH SET
                        rel.last_mentioned = $current_formatted_time,
                        rel.usage_count = COALESCE(rel.usage_count, 0) + 1{additional_set_properties_str}
                    RETURN n.name AS source, type(rel) AS relationship, m.name AS target
                    """
                params = {
                    "source_name": source,
                    "source_type": source_type,
                    "dest_name": destination,
                    "destination_type": destination_type,
                    "source_embedding": source_embedding,
                    "dest_embedding": dest_embedding,
                    "user_id": user_id,
                    "current_formatted_time": current_formatted_time,
                }

                # Add additional parameters to params if they exist
                if weight is not None:
                    params["weight"] = weight
                if is_uncertain is not None:
                    params["is_uncertain"] = is_uncertain
                if status is not None:
                    params["status"] = status
                if start_date is not None:
                    params["start_date"] = start_date
                if end_date is not None:
                    params["end_date"] = end_date
                if emotion is not None:
                    params["emotion"] = emotion
                if last_mentioned is not None:
                    params["last_mentioned"] = last_mentioned
                if usage_count is not None:
                    params["usage_count"] = usage_count

                logger.debug(f"Executing Cypher (neither exist): {cypher} with params: {params}")
                resp = self.graph.query(cypher, params=params)
                logger.debug(f"Graph query response: {resp}")
                results.append(resp)
        
        logger.debug(f"Finished adding entities. Results: {results}")
        return results

    def _remove_spaces_from_entities(self, entity_list):
        """
        Process entities by:
        1. Converting entity names to lowercase and replacing spaces with underscores
        2. Ensuring all required parameters are present with default values if missing
        """
        for item in entity_list:
            item["source"] = item["source"].lower().replace(" ", "_")
            item["relationship"] = item["relationship"].lower().replace(" ", "_")
            item["destination"] = item["destination"].lower().replace(" ", "_")

            # Ensure all required parameters are present with default values if missing
            if "weight" not in item or item["weight"] is None:
                item["weight"] = "relevant"  # Default to relevant significance

            if "is_uncertain" not in item or item["is_uncertain"] is None:
                item["is_uncertain"] = False  # Default to certain

            if "status" not in item or item["status"] is None:
                item["status"] = "active"  # Default to active status

            if "emotion" not in item or item["emotion"] is None:
                item["emotion"] = "neutral"  # Default to neutral emotion
                logger.info(f"Emotion defaulted to neutral: {item['source']} -> {item['relationship']} -> {item['destination']}")
            else:
                logger.debug(f"Emotion preserved: {item['emotion']} for {item['source']} -> {item['relationship']} -> {item['destination']}")

            if "last_mentioned" not in item or item["last_mentioned"] is None:
                item["last_mentioned"] = datetime.now(pytz.utc).isoformat()  # Default to current time in ISO format

            if "usage_count" not in item or item["usage_count"] is None:
                item["usage_count"] = 1  # Default to 1

            # Optional date parameters - no defaults for these
            # start_date and end_date can remain null

        return entity_list

    def _search_source_node(self, source_embedding, user_id, threshold=0.9):
        cypher = """
            MATCH (source_candidate)
            WHERE source_candidate.embedding IS NOT NULL 
            AND source_candidate.user_id = $user_id

            WITH source_candidate,
                round(
                    reduce(dot = 0.0, i IN range(0, size(source_candidate.embedding)-1) |
                        dot + source_candidate.embedding[i] * $source_embedding[i]) /
                    (sqrt(reduce(l2 = 0.0, i IN range(0, size(source_candidate.embedding)-1) |
                        l2 + source_candidate.embedding[i] * source_candidate.embedding[i])) *
                    sqrt(reduce(l2 = 0.0, i IN range(0, size($source_embedding)-1) |
                        l2 + $source_embedding[i] * $source_embedding[i])))
                , 4) AS source_similarity
            WHERE source_similarity >= $threshold

            WITH source_candidate, source_similarity
            ORDER BY source_similarity DESC
            LIMIT 1

            RETURN elementId(source_candidate)
            """

        params = {
            "source_embedding": source_embedding,
            "user_id": user_id,
            "threshold": threshold,
        }

        result = self.graph.query(cypher, params=params)
        return result

    def _search_destination_node(self, destination_embedding, user_id, threshold=0.9):
        cypher = """
            MATCH (destination_candidate)
            WHERE destination_candidate.embedding IS NOT NULL 
            AND destination_candidate.user_id = $user_id

            WITH destination_candidate,
                round(
                    reduce(dot = 0.0, i IN range(0, size(destination_candidate.embedding)-1) |
                        dot + destination_candidate.embedding[i] * $destination_embedding[i]) /
                    (sqrt(reduce(l2 = 0.0, i IN range(0, size(destination_candidate.embedding)-1) |
                        l2 + destination_candidate.embedding[i] * destination_candidate.embedding[i])) *
                    sqrt(reduce(l2 = 0.0, i IN range(0, size($destination_embedding)-1) |
                        l2 + $destination_embedding[i] * $destination_embedding[i])))
                , 4) AS destination_similarity
            WHERE destination_similarity >= $threshold

            WITH destination_candidate, destination_similarity
            ORDER BY destination_similarity DESC
            LIMIT 1

            RETURN elementId(destination_candidate)
            """
        params = {
            "destination_embedding": destination_embedding,
            "user_id": user_id,
            "threshold": threshold,
        }

        result = self.graph.query(cypher, params=params)
        return result

    def update_relationship(
        self, source, relationship, destination, user_id, **properties
    ):
        """
        Update a relationship with new properties.

        Args:
            source (str): Source node name
            relationship (str): Relationship type
            destination (str): Destination node name
            user_id (str): User ID
            **properties: Additional properties to update (weight, is_uncertain, status, start_date, end_date, emotion, last_mentioned, usage_count)

        Returns:
            dict: Updated relationship data
        """
        # Build the SET clause dynamically based on provided properties
        current_formatted_time_update = datetime.now(pytz.utc).strftime('%Y-%m-%d %H:%M:%S')
        set_clauses = [
            f"r.updated_at = $current_formatted_time_update"
        ]  # Use a distinct param name
        params = {
            "source_name": source,
            "dest_name": destination,
            "user_id": user_id,
            "current_formatted_time_update": current_formatted_time_update,  # Add to params
        }

        # Add each property to the SET clause if provided
        for key, value in properties.items():
            if value is not None:
                set_clauses.append(f"r.{key} = ${key}")
                params[key] = value

        # Create the query with the dynamic SET clause
        set_clause = ", ".join(set_clauses)
        cypher = f"""
        MATCH (n {{name: $source_name, user_id: $user_id}})
        -[r:{relationship}]->
        (m {{name: $dest_name, user_id: $user_id}})
        SET {set_clause}
        RETURN 
            n.name AS source,
            m.name AS target,
            type(r) AS relationship,
            r.weight AS weight,
            r.is_uncertain AS is_uncertain,
            r.status AS status,
            r.start_date AS start_date,
            r.end_date AS end_date,
            r.emotion AS emotion,
            r.last_mentioned AS last_mentioned,
            r.usage_count AS usage_count
        """

        result = self.graph.query(cypher, params=params)

        if result:
            result_dict = {
                "source": result[0]["source"],
                "relationship": result[0]["relationship"],
                "target": result[0]["target"],
            }

            # Add optional parameters if they exist in the result
            if result[0].get("weight") is not None:
                result_dict["weight"] = result[0]["weight"]
            if result[0].get("is_uncertain") is not None:
                result_dict["is_uncertain"] = result[0]["is_uncertain"]
            if result[0].get("status") is not None:
                result_dict["status"] = result[0]["status"]
            if result[0].get("start_date") is not None:
                result_dict["start_date"] = result[0]["start_date"]
            if result[0].get("end_date") is not None:
                result_dict["end_date"] = result[0]["end_date"]
            if result[0].get("emotion") is not None:
                result_dict["emotion"] = result[0]["emotion"]
            if result[0].get("last_mentioned") is not None:
                result_dict["last_mentioned"] = result[0]["last_mentioned"]
            if result[0].get("usage_count") is not None:
                result_dict["usage_count"] = result[0]["usage_count"]

            return result_dict

        return {"source": source, "relationship": relationship, "target": destination}

    def update_mention_metadata(self, memory_object, user_time, user_id, timezone_offset=None):
        """
        Update usage_count and last_mentioned for a memory object (node or relation) when it's selected for use.
        
        Args:
            memory_object (dict): Dictionary representing a node or relation with source, relationship, destination
            user_time (str): Current time in ISO format string (UTC, adjusted to user's local time)
            user_id (str): User ID for filtering
            timezone_offset (float, optional): Offset handling if needed separately
            
        Behavior:
            - If usage_count is not present → initialize to 1
            - If last_mentioned is not present → initialize to user_time
            - If both are present → increment usage_count and overwrite last_mentioned with user_time
        """
        # Extract relationship components
        source = memory_object.get("source")
        relationship = memory_object.get("relationship") or memory_object.get("relatationship")
        destination = memory_object.get("destination") or memory_object.get("target")
        
        if not all([source, relationship, destination]):
            logger.warning(f"Incomplete memory object for update: {memory_object}")
            return
            
        # Get current values or set defaults
        current_usage_count = memory_object.get("usage_count", 0)
        new_usage_count = current_usage_count + 1
        
        # Update the relationship with incremented usage_count and current timestamp
        update_cypher = """
        MATCH (n {name: $source_name, user_id: $user_id})
        -[r]->(m {name: $dest_name, user_id: $user_id})
        WHERE type(r) = $relationship_type
        SET r.usage_count = $new_usage_count,
            r.last_mentioned = $last_mentioned
        RETURN r.usage_count AS updated_count
        """
        
        params = {
            "source_name": source,
            "dest_name": destination,
            "user_id": user_id,
            "relationship_type": relationship,
            "new_usage_count": new_usage_count,
            "last_mentioned": user_time
        }
        
        try:
            result = self.graph.query(update_cypher, params=params)
            if result:
                logger.debug(f"Updated mention metadata: {source} -> {relationship} -> {destination} (count: {new_usage_count})")
            else:
                logger.warning(f"No relationship found to update: {source} -> {relationship} -> {destination}")
        except Exception as e:
            logger.error(f"Error updating mention metadata: {e}")

    def update_node_mention_metadata(self, node_name, user_time, user_id, timezone_offset=None):
        """
        Update usage_count and last_mentioned for a node when it's actively referenced.
        
        Args:
            node_name (str): Name of the node
            user_time (str): Current time in ISO format string (UTC, adjusted to user's local time)  
            user_id (str): User ID for filtering
            timezone_offset (float, optional): Offset handling if needed separately
        """
        # Update node metadata
        update_cypher = """
        MATCH (n {name: $node_name, user_id: $user_id})
        SET n.usage_count = COALESCE(n.usage_count, 0) + 1,
            n.last_mentioned = $last_mentioned
        RETURN n.usage_count AS updated_count
        """
        
        params = {
            "node_name": node_name,
            "user_id": user_id,
            "last_mentioned": user_time
        }
        
        try:
            result = self.graph.query(update_cypher, params=params)
            if result:
                logger.debug(f"Updated node mention metadata: {node_name} (count: {result[0]['updated_count']})")
            else:
                logger.warning(f"No node found to update: {node_name}")
        except Exception as e:
            logger.error(f"Error updating node mention metadata: {e}")

    def apply_weight_updates(self, user_id: str, updates: list, session_id: str = None):
        """
        Apply weight updates to nodes and relationships in the graph.
        
        This function is called by the backend system at the end of each session to update
        weight labels for graph entities based on cognitive analysis performed externally.
        
        Args:
            user_id (str): User ID for filtering entities
            updates (list): List of update dictionaries, each containing:
                - entity_id (str): The Neo4j elementId of the entity to update
                - type (str): Either "relation" or "node" 
                - weight_label (str): New weight label to apply (e.g. "important", "peripheral")
            session_id (str, optional): Session ID to track which session made these updates
            
        Returns:
            dict: Summary of updates applied, including counts of successful and failed updates
            
        Example:
            updates = [
                {"entity_id": "4:abc123:456", "type": "relation", "weight_label": "important"},
                {"entity_id": "4:def789:123", "type": "node", "weight_label": "peripheral"}
            ]
            result = graph.apply_weight_updates("user123", updates, "session456")
        """
        results = {
            "successful_updates": 0,
            "failed_updates": 0,
            "details": []
        }
        
        current_time = datetime.now(pytz.utc).isoformat()
        
        for i, update in enumerate(updates):
            entity_id = update.get("entity_id")
            entity_type = update.get("type")
            weight_label = update.get("weight_label")
            
            if not all([entity_id, entity_type, weight_label]):
                logger.warning(f"Incomplete update at index {i}: {update}")
                results["failed_updates"] += 1
                results["details"].append({
                    "entity_id": entity_id,
                    "status": "failed",
                    "reason": "missing_required_fields"
                })
                continue
            
            try:
                if entity_type == "relation":
                    # Update relationship weight
                    cypher = """
                    MATCH ()-[r]->()
                    WHERE elementId(r) = $entity_id AND r.user_id = $user_id
                    SET r.weight = $weight_label,
                        r.last_updated = $last_updated
                    """
                    
                    # Add session_id if provided
                    if session_id:
                        cypher += ", r.last_session_id = $session_id"
                    
                    cypher += """
                    RETURN elementId(r) AS entity_id,
                           type(r) AS relationship_type,
                           r.weight AS new_weight
                    """
                    
                    params = {
                        "entity_id": entity_id,
                        "user_id": user_id,
                        "weight_label": weight_label,
                        "last_updated": current_time
                    }
                    
                    if session_id:
                        params["session_id"] = session_id
                        
                elif entity_type == "node":
                    # Update node weight
                    cypher = """
                    MATCH (n)
                    WHERE elementId(n) = $entity_id AND n.user_id = $user_id
                    SET n.weight = $weight_label,
                        n.last_updated = $last_updated
                    """
                    
                    # Add session_id if provided
                    if session_id:
                        cypher += ", n.last_session_id = $session_id"
                        
                    cypher += """
                    RETURN elementId(n) AS entity_id,
                           n.name AS node_name,
                           n.weight AS new_weight
                    """
                    
                    params = {
                        "entity_id": entity_id,
                        "user_id": user_id,
                        "weight_label": weight_label,
                        "last_updated": current_time
                    }
                    
                    if session_id:
                        params["session_id"] = session_id
                        
                else:
                    logger.warning(f"Invalid entity type '{entity_type}' for entity {entity_id}")
                    results["failed_updates"] += 1
                    results["details"].append({
                        "entity_id": entity_id,
                        "status": "failed",
                        "reason": f"invalid_entity_type: {entity_type}"
                    })
                    continue
                
                # Execute the update query
                query_result = self.graph.query(cypher, params=params)
                
                if query_result:
                    results["successful_updates"] += 1
                    results["details"].append({
                        "entity_id": entity_id,
                        "status": "success",
                        "new_weight": weight_label,
                        "type": entity_type
                    })
                else:
                    results["failed_updates"] += 1
                    results["details"].append({
                        "entity_id": entity_id,
                        "status": "failed",
                        "reason": "entity_not_found"
                    })
                    
            except Exception as e:
                logger.error(f"Error updating entity {entity_id}: {str(e)}")
                results["failed_updates"] += 1
                results["details"].append({
                    "entity_id": entity_id,
                    "status": "failed",
                    "reason": f"error: {str(e)}"
                })
        
        return results

    def get_role_candidates(self, user_id, normalized_role, side="source"):
        """
        Retrieve candidate entities from the graph that match a given role relationship.
        
        Args:
            user_id (str): User ID for filtering
            normalized_role (str): The normalized role type (e.g., "best_friend", "sister", "roommate")
            side (str): Either "source" or "destination" - indicates which side of the relationship to query
            
        Returns:
            list: List of candidate dictionaries with keys:
                - name: The entity name
                - normalized_name: Lowercase version of the name
                - relationship_type: The relationship type that matched
                - element_id: Neo4j element ID of the entity
                - weight: Relationship weight/importance
                - status: Relationship status (active, ended, etc.)
                
        Example:
            get_role_candidates("user123", "best_friend", "source")
            # Returns people who have a "has_best_friend" relationship from USER_ID
        """
        # Map normalized roles to potential relationship types
        # This mapping can be extended dynamically or configured
        role_to_relationship_map = {
            "best_friend": ["has_best_friend", "is_best_friend_of", "friend_of"],
            "friend": ["friend_of", "is_friend_of", "friends_with"],
            "brother": ["has_brother", "brother_of", "has_sibling"],
            "sister": ["has_sister", "sister_of", "has_sibling"],
            "sibling": ["has_sibling", "sibling_of"],
            "mother": ["has_mother", "mother_of", "has_parent"],
            "father": ["has_father", "father_of", "has_parent"],
            "parent": ["has_parent", "parent_of"],
            "child": ["has_child", "child_of", "parent_of"],
            "son": ["has_son", "son_of", "has_child"],
            "daughter": ["has_daughter", "daughter_of", "has_child"],
            "roommate": ["lives_with", "roommate_of", "shares_apartment_with"],
            "colleague": ["works_with", "colleague_of", "coworker_of"],
            "manager": ["reports_to", "managed_by", "has_manager"],
            "partner": ["partner_of", "in_relationship_with", "dating"],
            "spouse": ["married_to", "spouse_of", "partner_of"],
            "boyfriend": ["dating", "boyfriend_of", "in_relationship_with"],
            "girlfriend": ["dating", "girlfriend_of", "in_relationship_with"],
            "neighbor": ["neighbor_of", "lives_near", "next_door_to"],
            "mentor": ["mentored_by", "has_mentor", "learns_from"],
            "student": ["teaches", "mentors", "has_student"],
            "boss": ["reports_to", "works_for", "employed_by"],
            "employee": ["employs", "manages", "supervises"],
        }
        
        # Get potential relationship types for this role
        relationship_types = role_to_relationship_map.get(normalized_role, [normalized_role])
        
        # Build query based on side
        if side == "source":
            # Query: USER_ID -[relationship]-> Person
            # Find entities where USER_ID has the role relationship TO them
            cypher_query = """
            MATCH (u {user_id: $user_id, name: $user_node_name})
            -[r]->(p)
            WHERE type(r) IN $relationship_types
            AND p.user_id = $user_id
            RETURN 
                p.name AS name,
                toLower(p.name) AS normalized_name,
                type(r) AS relationship_type,
                elementId(p) AS element_id,
                r.weight AS weight,
                r.status AS status,
                r.emotion AS emotion,
                r.last_mentioned AS last_mentioned
            ORDER BY r.last_mentioned DESC, r.usage_count DESC
            """
            params = {
                "user_id": user_id,
                "user_node_name": user_id,  # Assuming user node is named with user_id
                "relationship_types": relationship_types
            }
        else:  # side == "destination"
            # Query: Person -[relationship]-> USER_ID
            # Find entities that have the role relationship FROM them to USER_ID
            cypher_query = """
            MATCH (p)-[r]->(u {user_id: $user_id, name: $user_node_name})
            WHERE type(r) IN $relationship_types
            AND p.user_id = $user_id
            RETURN 
                p.name AS name,
                toLower(p.name) AS normalized_name,
                type(r) AS relationship_type,
                elementId(p) AS element_id,
                r.weight AS weight,
                r.status AS status,
                r.emotion AS emotion,
                r.last_mentioned AS last_mentioned
            ORDER BY r.last_mentioned DESC, r.usage_count DESC
            """
            params = {
                "user_id": user_id,
                "user_node_name": user_id,
                "relationship_types": relationship_types
            }
        
        try:
            results = self.graph.query(cypher_query, params=params)
            
            candidates = []
            for result in results:
                # Filter out inactive relationships by default
                status = result.get("status", "active")
                if status in ["active", None]:  # Include active and relationships without status
                    candidates.append({
                        "name": result["name"],
                        "normalized_name": result["normalized_name"],
                        "relationship_type": result["relationship_type"],
                        "element_id": result["element_id"],
                        "weight": result.get("weight"),
                        "status": status,
                        "emotion": result.get("emotion"),
                        "last_mentioned": result.get("last_mentioned")
                    })
            
            logger.info(f"Found {len(candidates)} candidates for role '{normalized_role}' (side={side})")
            return candidates
            
        except Exception as e:
            logger.error(f"Error querying role candidates for '{normalized_role}': {e}")
            return []

    def reset(self):
        """Reset the graph by clearing all nodes and relationships."""
        logger.warning("Clearing graph...")
        cypher_query = """
        MATCH (n) DETACH DELETE n
        """
        return self.graph.query(cypher_query)
    
    def close(self):
        """Clean up resources including thread pool executor."""
        logger.info("Closing MemoryGraph resources...")
        self._executor.shutdown(wait=False)
    
    def __del__(self):
        """Destructor to ensure thread pool is closed."""
        try:
            self.close()
        except:
            pass
