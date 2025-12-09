import json
import logging
import os
import uuid
from contextlib import contextmanager
from time import perf_counter
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
        self.trace_enabled = os.getenv("MEM0_TRACE", "1") != "0"
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

    def _log(self, level=logging.INFO, **fields):
        """Lightweight structured logging helper."""
        if not self.trace_enabled:
            return
        try:
            msg = " ".join(f"{k}={v}" for k, v in fields.items() if v is not None)
            logger.log(level, msg)
        except Exception:
            logger.exception("Failed to emit trace log", exc_info=True)

    @contextmanager
    def _log_span(self, event, level=logging.INFO, log_start=True, **fields):
        """
        Context manager to log start/end/duration for latency-sensitive operations.
        """
        if not self.trace_enabled:
            yield
            return
        start = perf_counter()
        if log_start:
            self._log(level, event=event, phase="start", **fields)
        try:
            yield
            duration_ms = round((perf_counter() - start) * 1000, 2)
            self._log(level, event=event, phase="end", duration_ms=duration_ms, **fields)
        except Exception as e:
            self._log(logging.ERROR, event=event, phase="error", error=str(e), **fields)
            raise

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

        with self._log_span(
            "llm_analyze_relation_evolution",
            user_id=user_id,
            trace_id=None,
            provider=self.llm_provider,
            relationship=current_relation.get("relatationship"),
            log_start=False,  # reduce noise; keep duration
        ):
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
        trace_id = filters.get("trace_id")
        with self._log_span(
            "add",
            user_id=filters.get("user_id"),
            trace_id=trace_id,
        ):
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
            # NOTE: delete/deactivation stage temporarily disabled for performance; re-enable later if needed.
            # original call:
            # to_be_updated = self._get_delete_entities_from_search_output(
            #     search_output, data, filters
            # )

            # TODO: Batch queries with APOC plugin
            # TODO: Add more filter support
            
            # Step 6: Process relationship updates
            # NOTE: delete/deactivation stage temporarily disabled for performance; re-enable later if needed.
            # original call:
            # updated_entities = self._process_relationship_updates(to_be_updated, filters["user_id"])
            updated_entities = []
            
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
        job_id = f"bg-{threading.get_ident()}"
        with self._log_span(
            "background_weight_adjustment",
            user_id=filters.get("user_id"),
            trace_id=filters.get("trace_id"),
            job_id=job_id,
            result_count=len(search_results),
        ):
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
        trace_id = filters.get("trace_id")
        with self._log_span(
            "search",
            user_id=filters.get("user_id"),
            trace_id=trace_id,
            limit=limit,
        ):
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
            with self._log_span(
                "bm25_rerank",
                user_id=filters.get("user_id"),
                trace_id=trace_id,
                candidate_count=len(search_outputs_sequence),
            ):
                reranked_results = bm25.get_top_n(tokenized_query, search_outputs_sequence, n=15)

            # Step 4: Build final results and update metadata
            search_results = []
            current_time_iso = datetime.now(pytz.utc).isoformat()
            relationship_updates = {}
            node_updates = {}
            
            for item in reranked_results:
                result_dict = None

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
                        break
                else:
                    # Fallback if original item not found
                    result_dict = {"source": item[0], "relationship": item[1], "destination": item[2]}

                search_results.append(result_dict)

                rel_key = (result_dict["source"], result_dict["relationship"], result_dict["destination"])
                relationship_updates[rel_key] = relationship_updates.get(rel_key, 0) + 1
                node_updates[result_dict["source"]] = node_updates.get(result_dict["source"], 0) + 1
                node_updates[result_dict["destination"]] = node_updates.get(result_dict["destination"], 0) + 1

            # Batch mention metadata updates to avoid per-item writes during search
            self.bulk_update_mention_metadata(relationship_updates, current_time_iso, filters["user_id"])
            self.bulk_update_node_mention_metadata(node_updates, current_time_iso, filters["user_id"])

            logger.info(f"Returned {len(search_results)} search results")

            # Trigger weight adjustment in the background without blocking
            # Pass the search query as context for better weight analysis
            # froze this below, cause it might be slowing down the funciton
            # self._executor.submit(
            #     self._background_weight_adjustment, 
            #     search_results.copy(),  # Copy to avoid modification issues
            #     query,  # Pass the search query as context
            #     filters
            # )

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
        relationship_updates = {}
        node_updates = {}
        
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

            rel_key = (result["source"], result["relationship"], result["target"])
            relationship_updates[rel_key] = relationship_updates.get(rel_key, 0) + 1
            node_updates[result["source"]] = node_updates.get(result["source"], 0) + 1
            node_updates[result["target"]] = node_updates.get(result["target"], 0) + 1

            final_results.append(result_dict)

        # Batch mention metadata updates to avoid per-item writes during bulk retrieval
        self.bulk_update_mention_metadata(relationship_updates, current_time_iso, filters["user_id"])
        self.bulk_update_node_mention_metadata(node_updates, current_time_iso, filters["user_id"])

        logger.info(f"Retrieved {len(final_results)} relationships")

        return final_results

    def _retrieve_nodes_from_data(self, data, filters):
        """Extracts all the entities mentioned in the query."""
        _tools = [EXTRACT_ENTITIES_TOOL]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [EXTRACT_ENTITIES_STRUCT_TOOL]

        with self._log_span(
            "llm_extract_entities",
            user_id=filters.get("user_id"),
            trace_id=filters.get("trace_id"),
            provider=self.llm_provider,
            log_start=False,  # only duration to reduce noise
        ):
            search_results = self.llm.generate_response(
                messages=[
                    {
                        "role": "system",
                        "content": f"""You are a smart assistant who understands entities and their types in a given text.

Pronoun Resolution Rules:
- First-person: If text contains 'I', 'me', 'my', 'myself' etc., use {filters['user_id']} as the entity
- Third-person: For 'he', 'she', 'they', 'him', 'her', 'them', resolve to the actual person mentioned in the context
- NEVER extract literal pronouns as entities. Do not create entities named 'i', 'me', 'my', 'he', 'she', 'they', 'him', 'her', 'them'
- Keep individual people separate - avoid composite names like 'john_mary'
- The primary user node ({filters['user_id']}) must always have entity_type 'person'.

Contextual Labeling Rules:
- Read the entire input and use every clue (roles, attributes, possessions, timing, setting) to infer the most fitting semantic category.
- Assign an entity_type for every entity. Only use 'unknown' if you can find no contextual signal even after considering the whole input.
- Use concise, single-word nouns that reflect the entity's nature (e.g., person, location, organization, object, product, vehicle, event, activity, time, concept, quantity). Stay consistent and do not invent hybrids or add parentheses.
- Let the entity's role in context guide the label: tangible things → object/product/vehicle; venues or physical settings → location/place; services or shops that users visit → location; actions or hobbies → activity; scheduled occurrences → event; dates, durations, or time expressions → time; abstract ideas or categories → concept.
- Prefer the label that best captures how the entity is being discussed in this specific input, rather than relying only on its literal wording.

Role Entity Rules:
- When a relational title is extracted (roommate, best_friend, coach, manager, teammate, barista, childhood_friend, colleague, neighbor, mentor, advisor, etc.), emit it as an entity with entity_type set to 'role'.
- Normalize to the base role label: lowercase with underscores for spaces (e.g., "Best Friend" → "best_friend", "Team Coach" → "team_coach").
- Do NOT attach temporal modifiers to role names. Strip them completely (e.g., "childhood friend" → extract "childhood" as separate time entity and "friend" as role entity; "former manager" → extract "former" as status/time indicator and "manager" as role entity).
- Role entities represent reusable anchors that may connect to multiple people. A functionally identical role mention (e.g., multiple roommate references) should resolve to the SAME role node name.
- Consistency is critical: if the user mentions "my roommate Alex" and later "my roommate Jordan", both should reference the entity "roommate" (not "roommate_alex" or "roommate_jordan").
- Temporal or stage-of-life modifiers must be separate entities with entity_type 'time' or 'concept', linked via their own relationships.

Extract all entities from the text with their types. ***DO NOT*** answer questions.""",
                },
                {"role": "user", "content": data},
            ],
            tools=_tools,
        )

        entity_type_map = {}

        try:
            raw_entities = []
            tool_calls = search_results.get("tool_calls") or []
            if tool_calls:
                raw_entities = tool_calls[0].get("arguments", {}).get("entities", [])
            else:
                raw_content = search_results.get("content")
                if raw_content:
                    try:
                        parsed = json.loads(raw_content)
                        raw_entities = parsed.get("entities", [])
                    except json.JSONDecodeError:
                        logger.warning(
                            "Failed to parse entity extractor content as JSON: %s",
                            raw_content,
                        )

            for item in raw_entities:
                entity = item.get("entity")
                entity_type = item.get("entity_type")
                if entity and entity_type:
                    entity_type_map[entity] = entity_type
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

        with self._log_span(
            "llm_extract_relations",
            user_id=filters.get("user_id"),
            trace_id=filters.get("trace_id"),
            provider=self.llm_provider,
            log_start=False,  # only duration to reduce noise
        ):
            extracted_entities = self.llm.generate_response(
                messages=messages,
                tools=_tools,
            )

        extracted_entities_list = []

        try:
            tool_calls = []
            if isinstance(extracted_entities, dict):
                tool_calls = extracted_entities.get("tool_calls") or []

            if tool_calls:
                first_call = tool_calls[0] if isinstance(tool_calls, list) else tool_calls
                if isinstance(first_call, dict):
                    extracted_entities_list = first_call.get("arguments", {}).get("entities") or []

            # Fallback: some LLMs may return entities directly in content instead of a tool call
            if not extracted_entities_list and isinstance(extracted_entities, dict):
                content = extracted_entities.get("content")
                content_json = None

                if isinstance(content, str):
                    try:
                        content_json = json.loads(content)
                    except json.JSONDecodeError:
                        content_json = None
                elif isinstance(content, dict):
                    content_json = content

                if isinstance(content_json, dict):
                    extracted_entities_list = content_json.get("entities") or []

            # Log emotions for debugging - using mem0 format
            for entity in extracted_entities_list:
                emotion = entity.get("emotion")
                logger.info(f"LLM extracted: {entity.get('source', '?')} -> {entity.get('relationship', '?')} -> {entity.get('destination', '?')}, emotion='{emotion}'")
        except Exception as e:
            logger.exception(
                f"Error parsing relations tool response: {e}, llm_provider={self.llm_provider}, extracted_entities={extracted_entities}"
            )

        extracted_entities = self._remove_spaces_from_entities(extracted_entities_list)
        
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

        with self._log_span(
            "neo4j_search_graph",
            user_id=filters.get("user_id"),
            trace_id=filters.get("trace_id"),
            node_count=len(node_list),
            threshold=self.threshold,
        ):
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

        with self._log_span(
            "llm_delete_graph_memory",
            user_id=filters.get("user_id"),
            trace_id=filters.get("trace_id"),
            provider=self.llm_provider,
            log_start=False,  # only duration to reduce noise
        ):
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
        """
        Add the new entities to the graph. Merge the nodes if they already exist.
        
        CRITICAL: All MERGE operations use {name, user_id} as the matching criteria.
        This ensures:
        1. Role nodes are reused within a user's graph (e.g., multiple "roommate → is → person" edges share one "roommate" node)
        2. Different users get independent role instances
        3. Consistent entity naming from the LLM prompts enables proper node reusability
        
        For example, when the LLM emits "roommate" for both Alex and Jordan, the MERGE finds
        the existing role node and adds a second "is" relationship, rather than creating duplicates.
        """
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

                # MERGE on {name, user_id} ensures role nodes are reused for this user
                # E.g., "roommate" for Alex and Jordan will find the same role node
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
                with self._log_span(
                    "neo4j_upsert_relationship",
                    user_id=user_id,
                    source=source,
                    destination=destination,
                    relationship=relationship,
                    log_start=False,
                ):
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

                # MERGE on {name, user_id} ensures role nodes are reused for this user
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
                with self._log_span(
                    "neo4j_upsert_relationship",
                    user_id=user_id,
                    source=source,
                    destination=destination,
                    relationship=relationship,
                    log_start=False,
                ):
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
                with self._log_span(
                    "neo4j_upsert_relationship",
                    user_id=user_id,
                    source=source,
                    destination=destination,
                    relationship=relationship,
                ):
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

                # MERGE on {name, user_id} ensures role nodes are reused for this user
                # Both source and destination can be role nodes that may already exist
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
                with self._log_span(
                    "neo4j_upsert_relationship",
                    user_id=user_id,
                    source=source,
                    destination=destination,
                    relationship=relationship,
                ):
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
        3. Filter out any literal pronoun nodes that slipped through
        """
        # Pronouns that should never become nodes
        PRONOUN_BLACKLIST = {'i', 'me', 'my', 'myself', 'he', 'she', 'they', 'him', 'her', 'them', 'his', 'hers', 'their', 'theirs'}
        
        filtered_entities = []
        for item in entity_list:
            item["source"] = item["source"].lower().replace(" ", "_")
            item["relationship"] = item["relationship"].lower().replace(" ", "_")
            item["destination"] = item["destination"].lower().replace(" ", "_")

            # Filter out relationships with pronoun nodes
            if item["source"] in PRONOUN_BLACKLIST or item["destination"] in PRONOUN_BLACKLIST:
                logger.warning(f"Filtered out pronoun relationship: {item['source']} -> {item['relationship']} -> {item['destination']}")
                continue

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

            # Add to filtered list if it passed all checks
            filtered_entities.append(item)

        return filtered_entities

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

        with self._log_span(
            "neo4j_update_relationship",
            user_id=user_id,
            source=source,
            destination=destination,
            relationship=relationship,
            log_start=False,
        ):
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

    def bulk_update_mention_metadata(self, relationship_updates, user_time, user_id, timezone_offset=None):
        """
        Batch relationship metadata updates to reduce per-item writes by updating usage counts and last_mentioned in a single UNWIND.
        """
        if not relationship_updates:
            return

        batch = [
            {
                "source": source,
                "relationship": relationship,
                "destination": destination,
                "increment": increment,
            }
            for (source, relationship, destination), increment in relationship_updates.items()
        ]

        update_cypher = """
        UNWIND $batch AS rel
        MATCH (n {name: rel.source, user_id: $user_id})-[r]->(m {name: rel.destination, user_id: $user_id})
        WHERE type(r) = rel.relationship
        SET r.usage_count = COALESCE(r.usage_count, 0) + rel.increment,
            r.last_mentioned = $last_mentioned
        RETURN count(r) AS updated_count
        """

        params = {
            "batch": batch,
            "user_id": user_id,
            "last_mentioned": user_time,
        }

        try:
            self.graph.query(update_cypher, params=params)
        except Exception as e:
            logger.error(f"Error batch updating mention metadata: {e}")

    def bulk_update_node_mention_metadata(self, node_updates, user_time, user_id, timezone_offset=None):
        """
        Batch node metadata updates to reduce per-item writes by updating usage counts and last_mentioned in a single UNWIND.
        """
        if not node_updates:
            return

        batch = [
            {"name": node_name, "increment": increment}
            for node_name, increment in node_updates.items()
        ]

        update_cypher = """
        UNWIND $batch AS node
        MATCH (n {name: node.name, user_id: $user_id})
        SET n.usage_count = COALESCE(n.usage_count, 0) + node.increment,
            n.last_mentioned = $last_mentioned
        RETURN count(n) AS updated_count
        """

        params = {
            "batch": batch,
            "user_id": user_id,
            "last_mentioned": user_time,
        }

        try:
            self.graph.query(update_cypher, params=params)
        except Exception as e:
            logger.error(f"Error batch updating node mention metadata: {e}")

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
        source = memory_object.get("source")
        relationship = memory_object.get("relationship") or memory_object.get("relatationship")
        destination = memory_object.get("destination") or memory_object.get("target")

        if not all([source, relationship, destination]):
            logger.warning(f"Incomplete memory object for update: {memory_object}")
            return

        # Delegate to bulk updater to keep all metadata writes consolidated
        self.bulk_update_mention_metadata(
            {(source, relationship, destination): 1},
            user_time,
            user_id,
            timezone_offset=timezone_offset,
        )

    def update_node_mention_metadata(self, node_name, user_time, user_id, timezone_offset=None):
        """
        Update usage_count and last_mentioned for a node when it's actively referenced.
        
        Args:
            node_name (str): Name of the node
            user_time (str): Current time in ISO format string (UTC, adjusted to user's local time)  
            user_id (str): User ID for filtering
            timezone_offset (float, optional): Offset handling if needed separately
        """
        # Delegate to bulk updater to keep all metadata writes consolidated
        self.bulk_update_node_mention_metadata(
            {node_name: 1},
            user_time,
            user_id,
            timezone_offset=timezone_offset,
        )

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
                with self._log_span(
                    "neo4j_apply_weight_update",
                    user_id=user_id,
                    entity_id=entity_id,
                    entity_type=entity_type,
                ):
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
