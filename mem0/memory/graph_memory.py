import logging
import json
from datetime import datetime
import pytz
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import uuid
import math

try:
    import numpy as _np
except ImportError:  # pragma: no cover - optional performance boost
    _np = None

from mem0.memory.utils import format_entities


class PersonDisambiguationException(Exception):
    """Base exception for person disambiguation issues."""
    pass


class AmbiguousPersonException(PersonDisambiguationException):
    """
    Raised when multiple existing person nodes match the new data equally well.
    
    Attributes:
        person_name: The name that's ambiguous
        candidates: List of candidate profiles that matched
        new_profile: The new person profile that couldn't be uniquely matched
        message: Human-readable error message
    """
    def __init__(self, person_name, candidates, new_profile, message=None):
        self.person_name = person_name
        self.candidates = candidates
        self.new_profile = new_profile
        self.message = message or f"Multiple existing nodes match '{person_name}'. User clarification needed."
        super().__init__(self.message)


class UnconfirmedPersonException(PersonDisambiguationException):
    """
    Raised when no existing person node overlaps with the new data.
    
    This suggests it might be a new person, but confirmation is needed before creating a new node.
    
    Attributes:
        person_name: The name in question
        existing_count: Number of existing nodes with this name
        new_profile: The new person profile
        message: Human-readable error message
    """
    def __init__(self, person_name, existing_count, new_profile, message=None):
        self.person_name = person_name
        self.existing_count = existing_count
        self.new_profile = new_profile
        self.message = message or f"No overlap found for '{person_name}' with {existing_count} existing node(s). Is this a new person?"
        super().__init__(self.message)

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

PRONOUN_BLACKLIST = {
    "me",
    "my",
    "myself",
    "he",
    "she",
    "they",
    "him",
    "her",
    "them",
    "his",
    "hers",
    "their",
    "theirs",
}


class MemoryGraph:
    LABEL_SIMILARITY_THRESHOLDS = {
        "person": 0.92,
        "location": 0.88,
        "organization": 0.9,
        "role": 0.9,
        "concept": 0.9,
        "default": 0.9,
    }
    NODE_SEARCH_CANDIDATE_LIMIT = 5
    CANDIDATE_NEIGHBOR_LIMIT = 10
    CANDIDATE_TIE_DELTA = 0.03
    HIGH_SIM_NEAR_THRESHOLD_DELTA = 0.02
    CONTEXT_MATCH_THRESHOLD = 0.85
    CONTEXT_MATCH_BONUS = 0.04
    CONTEXT_BUCKET_PENALTY = 0.05
    NO_CONTEXT_OVERLAP_SCALE = 0.5
    PROFILE_CONTRADICTION_PENALTY = 0.45

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

    def _get_label_threshold(self, label):
        """
        Returns the cosine similarity threshold for a specific label, defaulting to a global value.
        """
        if not label:
            return self.LABEL_SIMILARITY_THRESHOLDS["default"]
        return self.LABEL_SIMILARITY_THRESHOLDS.get(
            label.lower(), self.LABEL_SIMILARITY_THRESHOLDS["default"]
        )

    def _cosine_similarity(self, vec1, vec2):
        """Compute cosine similarity while guarding against zero vectors."""
        if not vec1 or not vec2:
            return 0.0
        length = min(len(vec1), len(vec2))
        if length == 0:
            return 0.0
        if _np is not None:
            v1 = _np.asarray(vec1[:length])
            v2 = _np.asarray(vec2[:length])
            denom = _np.linalg.norm(v1) * _np.linalg.norm(v2)
            if denom == 0:
                return 0.0
            return float(_np.dot(v1, v2) / denom)

        dot = 0.0
        norm1 = 0.0
        norm2 = 0.0
        for i in range(length):
            a = vec1[i]
            b = vec2[i]
            dot += a * b
            norm1 += a * a
            norm2 += b * b
        if not norm1 or not norm2:
            return 0.0
        return dot / (math.sqrt(norm1) * math.sqrt(norm2))

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
        
        Returns:
            dict: Dictionary with "updated_entities" and "added_entities" lists.
        
        Raises:
            AmbiguousPersonException: When multiple existing person nodes match the new data equally well.
                The exception contains candidates for user to choose from.
            UnconfirmedPersonException: When no existing person node overlaps with new data.
                The exception indicates confirmation is needed before creating a new person node.
        
        Note:
            These exceptions should be caught by the caller to prompt the user for disambiguation.
            For example:
                try:
                    result = memory_graph.add(data, filters)
                except AmbiguousPersonException as e:
                    # Ask user: "Which {e.person_name} do you mean: {describe candidates}?"
                except UnconfirmedPersonException as e:
                    # Ask user: "Is this a new person named {e.person_name}?"
        """
        # Step 1: Retrieve nodes from data
        entity_type_map = self._retrieve_nodes_from_data(data, filters)
        node_names = list(entity_type_map.keys())
        node_labels = [entity_type_map.get(name, "unknown") for name in node_names]
        
        # Step 2: Establish relations from data
        to_be_added = self._establish_nodes_relations_from_data(
            data, filters, entity_type_map
        )
        
        # Step 3: Search graph database
        search_output = self._search_graph_db(
            node_list=node_names, node_labels=node_labels, filters=filters
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
        
        # Step 7: Add entities (pass search_output for person profile matching)
        added_entities = self._add_entities(
            to_be_added, filters["user_id"], entity_type_map, search_output
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
        node_names = list(entity_type_map.keys())
        node_labels = [entity_type_map.get(name, "unknown") for name in node_names]
        
        # Step 2: Search graph database
        search_output = self._search_graph_db(
            node_list=node_names, node_labels=node_labels, filters=filters
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

                    # Add person_uid if present (for person nodes)
                    if orig_item.get("source_person_uid"):
                        result_dict["source_person_uid"] = orig_item["source_person_uid"]
                    if orig_item.get("destination_person_uid"):
                        result_dict["destination_person_uid"] = orig_item["destination_person_uid"]

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
            n.person_uid AS source_person_uid,
            labels(n) AS source_labels,
            type(r) AS relationship, 
            m.name AS target,
            m.person_uid AS target_person_uid,
            labels(m) AS target_labels,
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
            
            # Add person_uid if present (for person nodes)
            if result.get("source_person_uid"):
                result_dict["source_person_uid"] = result["source_person_uid"]
            if result.get("target_person_uid"):
                result_dict["target_person_uid"] = result["target_person_uid"]

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

- Role Entity Rules:
  - Whenever a relational title is mentioned (roommate, best_friend, coach, manager, teammate, barista, childhood_friend, colleague, neighbor, mentor, advisor, etc.), emit a dedicated entity with entity_type='role'.
  - This applies even if the role appears inside a compound subject (e.g., "my best friend Sibel and I…", "my coach Jordan and our team…"). Always create the reusable role entity in addition to the named person.
  - Normalize to the base role label: lowercase with underscores for spaces (e.g., "Best Friend" → "best_friend", "Team Coach" → "team_coach").
  - If the phrasing is "<person> is my <role>" (or "is our/their <role>"), treat it exactly the same: produce the role entity and the person entity.
  - Role detection checklist (apply every time a match is found):
    * Possessive phrases: "my/our/their <role> <name>", "<role> of mine/ours"
    * Reverse phrasing: "<name> is my/our/their <role>"
    * Appositives: "<name>, my/our/their <role>, …"
    * Coordinated subjects: "my/our/their <role> <name> and I …"
    If any of these patterns (or obvious variations) are present, you MUST emit both the role entity and the named person as separate entities.
  - Examples:
    * "My best friend Sibel and I tried a new class" → entities: {filters['user_id']} (person), best_friend (role), sibel (person), new_class (activity).
    * "Alex is my roommate" → entities: {filters['user_id']} (person), roommate (role), alex (person).
  - When you detect a role, prepare the graph for the two-hop structure (Owner → has → Role and Role → is → Person). Do NOT plan or output a direct relationship such as "owner → roommate → person"; that pattern is invalid given our schema.
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

        for entity in extracted_entities:
            owner_name = entity.get("owner_person_name")
            if isinstance(owner_name, str):
                owner_name = owner_name.strip()
                entity["owner_person_name"] = owner_name if owner_name else None
            elif owner_name is None:
                entity["owner_person_name"] = None
            else:
                entity["owner_person_name"] = None

        extracted_entities = self._remove_spaces_from_entities(extracted_entities)
        
        logger.debug(f"Extracted entities: {extracted_entities}")
        return extracted_entities

    def _search_graph_db(self, node_list, filters, node_labels=None, limit=100):
        """Search similar nodes and expand their relations with label-aware filtering."""
        if not node_list:
            return []

        node_labels = node_labels or []
        base_entries = []
        for idx, node in enumerate(node_list):
            label = node_labels[idx] if idx < len(node_labels) else None
            label_norm = label.lower() if isinstance(label, str) else None
            label_filter = label_norm if label_norm and label_norm != "unknown" else None
            base_entries.append(
                {
                    "name": node,
                    "label": label_norm,
                    "label_filter": label_filter,
                    "label_threshold": self._get_label_threshold(label_norm),
                    "global_threshold": self.threshold,
                    "embedding": self.embedding_model.embed(node),
                }
            )

        def _prepare_search_items(entries, use_label_filter):
            prepared = []
            for entry in entries:
                prepared.append(
                    {
                        "name": entry["name"],
                        "embedding": entry["embedding"],
                        "threshold": entry["label_threshold"] if use_label_filter else entry["global_threshold"],
                        "label_filter": entry["label_filter"] if use_label_filter else None,
                    }
                )
            return prepared

        def _execute_anchor_query(search_items, require_label_match):
            if not search_items:
                return []

            cypher_query = """
            UNWIND $search_items AS search_item
            MATCH (n)
            WHERE n.embedding IS NOT NULL 
              AND n.user_id = $user_id
            WITH search_item, n, [label IN labels(n) | toLower(label)] AS node_labels
            WHERE $require_label_match = false OR (
                search_item.label_filter IS NOT NULL AND search_item.label_filter IN node_labels
            )
            WITH search_item, n,
                 round(2 * vector.similarity.cosine(n.embedding, search_item.embedding) - 1, 4) AS similarity
            WHERE similarity >= search_item.threshold
            WITH search_item, n, similarity
            ORDER BY similarity DESC
            WITH search_item, collect({n: n, similarity: similarity})[..$limit] AS top_nodes
            UNWIND top_nodes AS top_node
            WITH search_item, top_node.n AS n, top_node.similarity AS similarity, $require_label_match AS require_label_match
            CALL (n) {
                MATCH (n)-[r]->(m)
                WHERE m.user_id = $user_id
                RETURN n.name AS source, elementId(n) AS source_id, labels(n) AS source_labels, n.person_uid AS source_person_uid,
                       type(r) AS relatationship, elementId(r) AS relation_id, 
                       m.name AS destination, elementId(m) AS destination_id, labels(m) AS destination_labels, m.person_uid AS destination_person_uid,
                       r.weight AS weight, r.is_uncertain AS is_uncertain, r.status AS status,
                       r.start_date AS start_date, r.end_date AS end_date, r.emotion AS emotion,
                       r.last_mentioned AS last_mentioned, r.usage_count AS usage_count
                UNION
                MATCH (m)-[r]->(n)
                WHERE m.user_id = $user_id
                RETURN m.name AS source, elementId(m) AS source_id, labels(m) AS source_labels, m.person_uid AS source_person_uid,
                       type(r) AS relatationship, elementId(r) AS relation_id,
                       n.name AS destination, elementId(n) AS destination_id, labels(n) AS destination_labels, n.person_uid AS destination_person_uid,
                       r.weight AS weight, r.is_uncertain AS is_uncertain, r.status AS status,
                       r.start_date AS start_date, r.end_date AS end_date, r.emotion AS emotion,
                       r.last_mentioned AS last_mentioned, r.usage_count AS usage_count
            }
            WITH DISTINCT search_item.name AS search_term, require_label_match,
                 source, source_id, source_labels, source_person_uid, relatationship, relation_id,
                 destination, destination_id, destination_labels, destination_person_uid, similarity,
                 weight, is_uncertain, status, start_date, end_date, emotion, last_mentioned, usage_count
            RETURN search_term, NOT require_label_match AS used_fallback,
                   source, source_id, source_labels, source_person_uid, relatationship, relation_id,
                   destination, destination_id, destination_labels, destination_person_uid, similarity,
                   weight, is_uncertain, status, start_date, end_date, emotion, last_mentioned, usage_count
            """

            params = {
                "search_items": search_items,
                "user_id": filters["user_id"],
                "limit": limit,
                "require_label_match": require_label_match,
            }
            return self.graph.query(cypher_query, params=params)

        label_entries = [entry for entry in base_entries if entry["label_filter"]]
        label_results = _execute_anchor_query(
            _prepare_search_items(label_entries, use_label_filter=True),
            require_label_match=True,
        )

        seen_terms = {row.get("search_term") for row in label_results if row.get("search_term")}
        fallback_targets = []
        fallback_target_names = set()
        for entry in base_entries:
            if entry["name"] not in seen_terms:
                fallback_targets.append(entry)
                fallback_target_names.add(entry["name"])

        # Log which entries had to fall back after attempting label-specific anchors
        for entry in label_entries:
            if entry["name"] in fallback_target_names:
                logger.info(
                    "[label_fallback] entity=%s label=%s threshold=%.3f",
                    entry["name"],
                    entry["label"],
                    entry["label_threshold"],
                )

        fallback_results = _execute_anchor_query(
            _prepare_search_items(fallback_targets, use_label_filter=False),
            require_label_match=False,
        )

        deduped = []
        seen_rel_ids = set()
        for record in label_results + fallback_results:
            key = (
                record.get("relation_id"),
                record.get("source_id"),
                record.get("destination_id"),
            )
            if key in seen_rel_ids:
                continue
            seen_rel_ids.add(key)
            deduped.append(record)

        return deduped

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

    def _add_entities(self, to_be_added, user_id, entity_type_map, search_output):
        """
        Add the new entities to the graph. Merge the nodes if they already exist.
        
        CRITICAL: All MERGE operations use {name, user_id} as the matching criteria.
        This ensures:
        1. Role nodes are reused within a user's graph (e.g., multiple "roommate → is → person" edges share one "roommate" node)
        2. Different users get independent role instances
        3. Consistent entity naming from the LLM prompts enables proper node reusability
        
        For person nodes specifically, we use person_uid to disambiguate between multiple people with the same name.
        The person identifier system:
        1. Builds profiles of existing person nodes based on their relationships
        2. Builds profiles of new person mentions from to_be_added
        3. Compares profiles to decide whether to reuse an existing node or create a new one
        4. Assigns person_uid when creating new person nodes
        
        For example, when the LLM emits "roommate" for both Alex and Jordan, the MERGE finds
        the existing role node and adds a second "is" relationship, rather than creating duplicates.
        """
        results = []
        logger.debug(f"Adding entities. `to_be_added`: {to_be_added}")

        embedding_cache = {}

        def _get_embedding(term):
            if term not in embedding_cache:
                embedding_cache[term] = self.embedding_model.embed(term)
            return embedding_cache[term]
        
        # Build existing person profiles from search output
        existing_person_profiles = self._build_person_profiles(search_output, entity_type_map)
        logger.debug(f"Existing person profiles: {list(existing_person_profiles.keys())}")
        
        # Build new person profiles from to_be_added using the same flattened structure
        # This ensures consistency with existing profiles built from search_output
        new_person_profiles = {}
        for item in to_be_added:
            source = item["source"]
            destination = item["destination"]
            relationship = item["relationship"]
            
            source_type = entity_type_map.get(source, "unknown")
            dest_type = entity_type_map.get(destination, "unknown")
            
            # Build profile for source if it's a person
            if source_type == "person":
                if source not in new_person_profiles:
                    new_person_profiles[source] = {
                        "person_uid": None,  # Will be assigned if we create a new node
                        "element_id": None,
                        "name": source,
                        "facts": []  # Flattened list of fact records
                    }
                
                # Build outgoing fact record
                target_embedding = None
                if destination and destination not in PRONOUN_BLACKLIST:
                    target_embedding = _get_embedding(destination)

                fact_record = {
                    "direction": "out",
                    "relationship": relationship,
                    "target_name": destination,
                    "target_uid": None,  # Will be filled if destination is also a new person with UID
                    "target_type": dest_type,
                    "target_label": dest_type,
                    "target_embedding": target_embedding,
                    "metadata": {
                        "weight": item.get("weight"),
                        "is_uncertain": item.get("is_uncertain"),
                        "status": item.get("status"),
                        "emotion": item.get("emotion"),
                        "start_date": item.get("start_date"),
                        "end_date": item.get("end_date"),
                    }
                }
                new_person_profiles[source]["facts"].append(fact_record)
            
            # Build profile for destination if it's a person
            if dest_type == "person":
                if destination not in new_person_profiles:
                    new_person_profiles[destination] = {
                        "person_uid": None,
                        "element_id": None,
                        "name": destination,
                        "facts": []
                    }
                
                # Build incoming fact record
                target_embedding = None
                if source and source not in PRONOUN_BLACKLIST:
                    target_embedding = _get_embedding(source)

                fact_record = {
                    "direction": "in",
                    "relationship": relationship,
                    "target_name": source,
                    "target_uid": None,  # Will be filled if source is also a new person with UID
                    "target_type": source_type,
                    "target_label": source_type,
                    "target_embedding": target_embedding,
                    "metadata": {
                        "weight": item.get("weight"),
                        "is_uncertain": item.get("is_uncertain"),
                        "status": item.get("status"),
                        "emotion": item.get("emotion"),
                        "start_date": item.get("start_date"),
                        "end_date": item.get("end_date"),
                    }
                }
                new_person_profiles[destination]["facts"].append(fact_record)
        
        logger.debug(f"New person profiles to add: {list(new_person_profiles.keys())}")
        
        # Build person resolution cache: for each person name, decide reuse/new/ambiguous
        # Also pre-generate person_uid for new nodes to ensure consistency within the batch
        person_decisions = {}
        person_uid_cache = {}  # person_name -> UUID for new nodes
        user_id_normalized = None
        if user_id:
            user_id_normalized = user_id.lower().replace(" ", "_")

        def _resolve_owner_person_uid(
            owner_name,
            source_name,
            destination_name,
            source_type_name,
            destination_type_name,
            source_uid,
            destination_uid,
        ):
            """
            Determine the owner_person_uid for a relationship using only the supplied owner name.
            """
            if not owner_name:
                return None
            if (
                owner_name == source_name
                and source_type_name == "person"
                and source_uid
            ):
                return source_uid
            if (
                owner_name == destination_name
                and destination_type_name == "person"
                and destination_uid
            ):
                return destination_uid
            cached_uid = person_uid_cache.get(owner_name)
            if cached_uid:
                return cached_uid
            lookup_uid = self._lookup_person_uid_by_name(owner_name, user_id)
            if lookup_uid:
                person_uid_cache[owner_name] = lookup_uid
            return lookup_uid

        for person_name, new_profile in new_person_profiles.items():
            existing_profiles_for_name = existing_person_profiles.get(person_name, [])
            skip_disambiguation_for_user = (
                user_id_normalized is not None and person_name == user_id_normalized
            )
            context_nodes = []
            for fact in new_profile.get("facts", []):
                target_name = fact.get("target_name")
                if not target_name:
                    continue
                labels = []
                target_label = fact.get("target_label") or fact.get("target_type")
                if target_label:
                    labels.append(target_label)
                context_nodes.append(
                    {
                        "name": target_name,
                        "labels": labels,
                        "embedding": fact.get("target_embedding"),
                        "relationship": fact.get("relationship"),
                        "direction": fact.get("direction"),
                    }
                )
            
            if skip_disambiguation_for_user:
                if existing_profiles_for_name:
                    primary_profile = existing_profiles_for_name[0]
                    decision = {
                        "decision": "reuse",
                        "element_id": primary_profile.get("element_id"),
                        "person_uid": primary_profile.get("person_uid"),
                        "matched_profile": primary_profile,
                        "all_comparisons": [],
                    }
                    logger.debug(
                        "Skipping disambiguation for user '%s' and reusing existing profile with person_uid=%s",
                        person_name,
                        primary_profile.get("person_uid"),
                    )
                else:
                    decision = {
                        "decision": "new",
                        "element_id": None,
                        "person_uid": None,
                        "matched_profile": None,
                        "all_comparisons": [],
                    }
                    logger.debug(
                        "Skipping disambiguation for user '%s' and creating a new profile",
                        person_name,
                    )
            else:
                existing_count = self._count_person_nodes(person_name, user_id)

                if existing_count == 0:
                    # No node with this name – create new without heavy disambiguation
                    decision = {
                        "decision": "new",
                        "element_id": None,
                        "person_uid": None,
                        "matched_profile": None,
                        "all_comparisons": [],
                    }
                elif existing_count == 1:
                    # Exactly one existing node – compare profiles to decide reuse vs new
                    base_profile = None
                    if len(existing_profiles_for_name) == 1:
                        base_profile = existing_profiles_for_name[0]
                    if base_profile is None:
                        base_profile = self._fetch_single_person_profile(
                            person_name, user_id
                        )
                        if base_profile:
                            existing_profiles_for_name = [base_profile]

                    comparison_result = None
                    comparisons = []
                    if base_profile:
                        comparison_result = self._compare_person_profiles(
                            base_profile, new_profile
                        )
                        comparisons.append((base_profile, comparison_result))

                    if comparison_result == "match":
                        decision = {
                            "decision": "reuse",
                            "element_id": base_profile.get("element_id") if base_profile else None,
                            "person_uid": base_profile.get("person_uid") if base_profile else None,
                            "matched_profile": base_profile,
                            "all_comparisons": comparisons,
                        }
                    elif comparison_result == "contradict":
                        decision = {
                            "decision": "new",
                            "element_id": None,
                            "person_uid": None,
                            "matched_profile": None,
                            "all_comparisons": comparisons,
                        }
                    else:
                        # No overlap or no profile info – create a new person to avoid wrong reuse
                        decision = {
                            "decision": "new",
                            "element_id": None,
                            "person_uid": None,
                            "matched_profile": None,
                            "all_comparisons": comparisons,
                        }
                        logger.info(
                            "[person_new_no_overlap] name=%s reason=%s",
                            person_name,
                            comparison_result or "no_profile",
                        )
                else:
                    # existing_count >= 2, run full disambiguation
                    person_embedding = _get_embedding(person_name)
                    person_candidates = self._search_source_node(
                        person_embedding,
                        user_id,
                        label="person",
                        limit=self.NODE_SEARCH_CANDIDATE_LIMIT,
                    )

                    profile_comparisons = {}
                    for profile in existing_profiles_for_name:
                        element_id = profile.get("element_id")
                        if not element_id:
                            continue
                        profile_comparisons[element_id] = self._compare_person_profiles(
                            profile, new_profile
                        )

                    candidate_neighbor_map = self._get_candidate_neighbor_names(
                        [candidate["node_id"] for candidate in person_candidates],
                        user_id,
                    )

                    scoring_result = self._score_entity_candidates(
                        {"name": person_name, "type": "person"},
                        person_candidates,
                        metadata={
                            "context_nodes": context_nodes,
                            "candidate_neighbors": candidate_neighbor_map,
                            "profile_comparisons": profile_comparisons,
                            "tie_delta": 0.04,
                        },
                    )
                    best_candidate = scoring_result.get("candidate")
                    best_score = scoring_result.get("score", 0.0)
                    tie = scoring_result.get("tie", False)
                    overlap_labels = scoring_result.get("overlap_labels") or []
                    penalties_applied = scoring_result.get("penalties_applied", False)
                    context_bucket_count = scoring_result.get("context_bucket_count", 0)
                    score_entries = scoring_result.get("scores", [])
                    best_details = score_entries[0] if score_entries else {}
                    base_similarity = best_details.get("similarity", 0.0)
                    penalty_reasons = set()
                    for penalty in best_details.get("penalties", []):
                        reason = penalty.get("reason") or penalty.get("type")
                        if reason:
                            penalty_reasons.add(reason)
                    context_penalty_reasons = {"no_overlap", "no_context_overlap"}
                    context_penalties_only = (
                        not penalty_reasons
                        or penalty_reasons.issubset(context_penalty_reasons)
                    )
                    matched_profile = None
                    if best_candidate:
                        matched_profile = next(
                            (
                                profile
                                for profile in existing_profiles_for_name
                                if profile.get("element_id") == best_candidate.get("node_id")
                            ),
                            None,
                        )
                    profile_result = None
                    if best_candidate:
                        profile_result = profile_comparisons.get(
                            best_candidate.get("node_id")
                        )

                    person_threshold = self._get_label_threshold("person")
                    has_context_overlap = bool(overlap_labels)
                    allow_reuse = has_context_overlap or profile_result == "match"
                    near_threshold = (
                        best_candidate
                        and best_score >= person_threshold - self.HIGH_SIM_NEAR_THRESHOLD_DELTA
                    )
                    penalties_triggered = penalties_applied or (
                        context_bucket_count > 0 and not has_context_overlap
                    )
                    single_existing_profile = len(existing_profiles_for_name) == 1
                    allow_reuse_without_overlap = (
                        best_candidate
                        and not allow_reuse
                        and not has_context_overlap
                        and single_existing_profile
                        and matched_profile is not None
                        and profile_result != "contradict"
                        and context_penalties_only
                        and (
                            best_score >= person_threshold
                            or base_similarity >= person_threshold
                        )
                    )
                    if tie:
                        decision = {
                            "decision": "ambiguous",
                            "element_id": None,
                            "person_uid": None,
                            "matched_profile": None,
                            "all_comparisons": scoring_result.get("scores", []),
                            "candidates": person_candidates,
                        }
                    elif best_candidate and best_score >= person_threshold and allow_reuse:
                        decision = {
                            "decision": "reuse",
                            "element_id": best_candidate.get("node_id"),
                            "person_uid": best_candidate.get("person_uid"),
                            "matched_profile": matched_profile,
                            "all_comparisons": scoring_result.get("scores", []),
                        }
                    elif best_candidate and allow_reuse_without_overlap:
                        logger.info(
                            "[person_reuse_fallback] name=%s reason=single_profile_no_overlap",
                            person_name,
                        )
                        decision = {
                            "decision": "reuse",
                            "element_id": best_candidate.get("node_id"),
                            "person_uid": best_candidate.get("person_uid"),
                            "matched_profile": matched_profile,
                            "all_comparisons": scoring_result.get("scores", []),
                        }
                    elif best_candidate and best_score >= person_threshold and not allow_reuse:
                        decision = {
                            "decision": "unconfirmed",
                            "element_id": None,
                            "person_uid": None,
                            "matched_profile": None,
                            "all_comparisons": scoring_result.get("scores", []),
                            "existing_count": existing_count,
                            "candidates": person_candidates,
                        }
                    elif best_candidate and near_threshold and penalties_triggered:
                        decision = {
                            "decision": "unconfirmed",
                            "element_id": None,
                            "person_uid": None,
                            "matched_profile": None,
                            "all_comparisons": scoring_result.get("scores", []),
                            "existing_count": existing_count,
                            "candidates": person_candidates,
                        }
                    elif best_candidate or existing_profiles_for_name:
                        decision = {
                            "decision": "unconfirmed",
                            "element_id": None,
                            "person_uid": None,
                            "matched_profile": None,
                            "all_comparisons": scoring_result.get("scores", []),
                            "existing_count": existing_count,
                            "candidates": person_candidates,
                        }
                    else:
                        decision = {
                            "decision": "new",
                            "element_id": None,
                            "person_uid": None,
                            "matched_profile": None,
                            "all_comparisons": scoring_result.get("scores", []),
                        }
            
            person_decisions[person_name] = decision
            
            # Handle ambiguous and unconfirmed cases
            # These require user interaction to resolve properly, except for the account owner
            if not skip_disambiguation_for_user:
                if decision["decision"] == "ambiguous":
                    logger.error(f"Ambiguous person match for '{person_name}'. Multiple candidates found.")
                    raise AmbiguousPersonException(
                        person_name=person_name,
                        candidates=decision.get("candidates", []),
                        new_profile=new_profile,
                        message=f"Cannot disambiguate '{person_name}': {len(decision.get('candidates', []))} existing nodes match equally. Please specify which person you mean."
                    )
                elif decision["decision"] == "unconfirmed":
                    logger.warning(f"Unconfirmed person match for '{person_name}'. No clear overlap with {decision.get('existing_count', 0)} existing node(s).")
                    raise UnconfirmedPersonException(
                        person_name=person_name,
                        existing_count=decision.get("existing_count", 0),
                        new_profile=new_profile,
                        message=f"Is '{person_name}' a new person? {decision.get('existing_count', 0)} existing node(s) found but no relationship overlap detected."
                    )
            
            # If decision is "new", generate and cache the person_uid now
            # This ensures all relationships for the same new person use the same UUID
            if decision["decision"] == "new":
                person_uid_cache[person_name] = str(uuid.uuid4())
                logger.debug(f"Pre-generated person_uid={person_uid_cache[person_name]} for new person '{person_name}'")
            elif decision["decision"] == "reuse":
                # Cache the existing person_uid for reused nodes too
                person_uid_cache[person_name] = decision["person_uid"]
        
        logger.debug(f"Person decisions: {person_decisions}")
        logger.debug(f"Person UID cache: {person_uid_cache}")
        
        # Now that person_uid_cache is complete, propagate UIDs into fact records
        # This allows more precise matching when both source and target are persons in this batch
        for person_name, profile in new_person_profiles.items():
            for fact in profile["facts"]:
                target_name = fact["target_name"]
                # If target is a person and we have a UID for them, fill it in
                if fact["target_type"] == "person" and target_name in person_uid_cache:
                    fact["target_uid"] = person_uid_cache[target_name]
                    logger.debug(f"Propagated target_uid for {person_name} -> {target_name}: {fact['target_uid']}")
        
        for i, item in enumerate(to_be_added):
            # entities
            source = item["source"]
            destination = item["destination"]
            relationship = item["relationship"]

            # types
            source_type = entity_type_map.get(source, "unknown")
            destination_type = entity_type_map.get(destination, "unknown")

            # embeddings
            source_embedding = _get_embedding(source)
            dest_embedding = _get_embedding(destination)

            # additional parameters
            weight = item.get("weight")
            is_uncertain = item.get("is_uncertain")
            status = item.get("status")
            start_date = item.get("start_date")
            end_date = item.get("end_date")
            emotion = item.get("emotion")
            last_mentioned = item.get("last_mentioned")
            usage_count = item.get("usage_count")
            owner_person_name = item.get("owner_person_name") or None
            owner_person_label = owner_person_name
            owner_person_uid = None

            def _build_relationship_properties(alias):
                clauses = []
                rel_params = {"owner_person_uid": owner_person_uid}
                clauses.append(f"{alias}.owner_person_uid = $owner_person_uid")
                if weight is not None:
                    clauses.append(f"{alias}.weight = $weight")
                    rel_params["weight"] = weight
                if is_uncertain is not None:
                    clauses.append(f"{alias}.is_uncertain = $is_uncertain")
                    rel_params["is_uncertain"] = is_uncertain
                if status is not None:
                    clauses.append(f"{alias}.status = $status")
                    rel_params["status"] = status
                if start_date is not None:
                    clauses.append(f"{alias}.start_date = $start_date")
                    rel_params["start_date"] = start_date
                if end_date is not None:
                    clauses.append(f"{alias}.end_date = $end_date")
                    rel_params["end_date"] = end_date
                if emotion is not None:
                    clauses.append(f"{alias}.emotion = $emotion")
                    rel_params["emotion"] = emotion
                if last_mentioned is not None:
                    clauses.append(f"{alias}.last_mentioned = $last_mentioned")
                    rel_params["last_mentioned"] = last_mentioned
                if usage_count is not None:
                    clauses.append(f"{alias}.usage_count = $usage_count")
                    rel_params["usage_count"] = usage_count

                additional_set_properties_str = ""
                if clauses:
                    additional_set_properties_str = ", " + ", ".join(clauses)
                return additional_set_properties_str, rel_params

            # For person nodes, use the decision from person_decisions instead of embedding search
            # This enables proper person disambiguation
            source_node_search_result = None
            destination_node_search_result = None
            source_person_uid_to_use = None
            dest_person_uid_to_use = None
            source_best_score = 0.0
            dest_best_score = 0.0
            
            if source_type == "person" and source in person_decisions:
                decision = person_decisions[source]
                if decision["decision"] == "reuse":
                    source_node_search_result = [{"elementId(source_candidate)": decision["element_id"]}]
                    source_person_uid_to_use = decision["person_uid"]
                    logger.debug(f"Reusing existing person node for source '{source}' with person_uid={source_person_uid_to_use}")
                elif decision["decision"] == "new":
                    # Use the cached person_uid to ensure consistency across all relationships in this batch
                    source_person_uid_to_use = person_uid_cache.get(source)
                    logger.debug(f"Creating new person node for source '{source}' with cached person_uid={source_person_uid_to_use}")
                    # source_node_search_result remains None to trigger the "new node" branch
            else:
                source_candidates = self._search_source_node(
                    source_embedding,
                    user_id,
                    label=source_type,
                    limit=self.NODE_SEARCH_CANDIDATE_LIMIT,
                )
                candidate_neighbor_map = self._get_candidate_neighbor_names(
                    [candidate["node_id"] for candidate in source_candidates],
                    user_id,
                )
                source_scoring = self._score_entity_candidates(
                    {"name": source, "type": source_type},
                    source_candidates,
                    metadata={
                        "context_nodes": [
                            {
                                "name": destination,
                                "labels": [destination_type],
                                "embedding": dest_embedding,
                            }
                        ],
                        "candidate_neighbors": candidate_neighbor_map,
                        "tie_delta": self.CANDIDATE_TIE_DELTA,
                    },
                )
                source_best_candidate = source_scoring.get("candidate")
                source_best_score = source_scoring.get("score", 0.0)
                source_threshold = self._get_label_threshold(source_type)
                if source_best_candidate and source_best_score >= source_threshold:
                    source_node_search_result = [
                        {"elementId(source_candidate)": source_best_candidate["node_id"]}
                    ]
                else:
                    if source_best_candidate and source_best_score >= source_threshold - self.HIGH_SIM_NEAR_THRESHOLD_DELTA:
                        logger.info(
                            "[candidate_rejected] entity=%s label=%s score=%.4f threshold=%.4f",
                            source,
                            source_type,
                            source_best_score,
                            source_threshold,
                        )
            
            if destination_type == "person" and destination in person_decisions:
                decision = person_decisions[destination]
                if decision["decision"] == "reuse":
                    destination_node_search_result = [{"elementId(destination_candidate)": decision["element_id"]}]
                    dest_person_uid_to_use = decision["person_uid"]
                    logger.debug(f"Reusing existing person node for destination '{destination}' with person_uid={dest_person_uid_to_use}")
                elif decision["decision"] == "new":
                    # Use the cached person_uid
                    dest_person_uid_to_use = person_uid_cache.get(destination)
                    logger.debug(f"Creating new person node for destination '{destination}' with cached person_uid={dest_person_uid_to_use}")
            else:
                destination_candidates = self._search_destination_node(
                    dest_embedding,
                    user_id,
                    label=destination_type,
                    limit=self.NODE_SEARCH_CANDIDATE_LIMIT,
                )
                destination_neighbor_map = self._get_candidate_neighbor_names(
                    [candidate["node_id"] for candidate in destination_candidates],
                    user_id,
                )
                destination_scoring = self._score_entity_candidates(
                    {"name": destination, "type": destination_type},
                    destination_candidates,
                    metadata={
                        "context_nodes": [
                            {
                                "name": source,
                                "labels": [source_type],
                                "embedding": source_embedding,
                            }
                        ],
                        "candidate_neighbors": destination_neighbor_map,
                        "tie_delta": self.CANDIDATE_TIE_DELTA,
                    },
                )
                destination_best_candidate = destination_scoring.get("candidate")
                dest_best_score = destination_scoring.get("score", 0.0)
                dest_threshold = self._get_label_threshold(destination_type)
            if destination_best_candidate and dest_best_score >= dest_threshold:
                destination_node_search_result = [
                    {"elementId(destination_candidate)": destination_best_candidate["node_id"]}
                ]
            else:
                    if destination_best_candidate and dest_best_score >= dest_threshold - self.HIGH_SIM_NEAR_THRESHOLD_DELTA:
                        logger.info(
                            "[candidate_rejected] entity=%s label=%s score=%.4f threshold=%.4f",
                            destination,
                            destination_type,
                            dest_best_score,
                        dest_threshold,
                    )

            owner_person_uid = _resolve_owner_person_uid(
                owner_person_name,
                source,
                destination,
                source_type,
                destination_type,
                source_person_uid_to_use,
                dest_person_uid_to_use,
            )
            if owner_person_name and owner_person_uid is None:
                logger.debug(
                    "Owner name '%s' supplied but no matching person_uid found for relationship %s -> %s -> %s",
                    owner_person_name,
                    source,
                    relationship,
                    destination,
                )
            if owner_person_uid:
                item["owner_person_name"] = owner_person_uid
            else:
                item["owner_person_name"] = owner_person_label

            logger.debug(f"Processing item: {item}. Source found: {bool(source_node_search_result)}. Destination found: {bool(destination_node_search_result)}")

            # TODO: Create a cypher query and common params for all the cases
            if not destination_node_search_result and source_node_search_result:
                current_formatted_time = datetime.now(pytz.utc).strftime('%Y-%m-%d %H:%M:%S')
                additional_set_properties_str, rel_params = _build_relationship_properties("r")

                # For person nodes with new person_uid, include it in MERGE to ensure distinct nodes
                # For other nodes or existing persons, use standard MERGE on (name, user_id)
                
                if destination_type == "person" and dest_person_uid_to_use and destination_node_search_result is None:
                    # New person node - MERGE on (name, user_id, person_uid) to ensure uniqueness
                    cypher = f"""
                        MATCH (source)
                        WHERE elementId(source) = $source_id
                        MERGE (destination:{destination_type} {{name: $destination_name, user_id: $user_id, person_uid: $dest_person_uid}})
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
                        "source_id": source_node_search_result[0]["elementId(source_candidate)"],
                        "destination_name": destination,
                        "relationship": relationship,
                        "destination_type": destination_type,
                        "destination_embedding": dest_embedding,
                        "user_id": user_id,
                        "current_formatted_time": current_formatted_time,
                        "dest_person_uid": dest_person_uid_to_use
                    }
                else:
                    # Non-person or existing person - standard MERGE on (name, user_id)
                    on_create_clauses = [
                        "destination.created_at = $current_formatted_time",
                        "destination.embedding = $destination_embedding"
                    ]
                    # Note: person_uid not in MERGE pattern, so don't set it here
                    
                    cypher = f"""
                        MATCH (source)
                        WHERE elementId(source) = $source_id
                        MERGE (destination:{destination_type} {{name: $destination_name, user_id: $user_id}})
                        ON CREATE SET
                            {", ".join(on_create_clauses)}
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
                        "source_id": source_node_search_result[0]["elementId(source_candidate)"],
                        "destination_name": destination,
                        "relationship": relationship,
                        "destination_type": destination_type,
                        "destination_embedding": dest_embedding,
                        "user_id": user_id,
                        "current_formatted_time": current_formatted_time,
                    }
                params.update(rel_params)

                logger.debug(f"Executing Cypher (source exists): {cypher} with params: {params}")
                resp = self.graph.query(cypher, params=params)
                logger.debug(f"Graph query response: {resp}")
                results.append(resp)

            elif destination_node_search_result and not source_node_search_result:
                current_formatted_time = datetime.now(pytz.utc).strftime('%Y-%m-%d %H:%M:%S')
                additional_set_properties_str, rel_params = _build_relationship_properties("r")

                # For person nodes with new person_uid, include it in MERGE to ensure distinct nodes
                # For other nodes or existing persons, use standard MERGE on (name, user_id)
                
                if source_type == "person" and source_person_uid_to_use and source_node_search_result is None:
                    # New person node - MERGE on (name, user_id, person_uid) to ensure uniqueness
                    cypher = f"""
                        MATCH (destination)
                        WHERE elementId(destination) = $destination_id
                        MERGE (source:{source_type} {{name: $source_name, user_id: $user_id, person_uid: $source_person_uid}})
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
                        "destination_id": destination_node_search_result[0]["elementId(destination_candidate)"],
                        "source_name": source,
                        "relationship": relationship,
                        "source_type": source_type,
                        "source_embedding": source_embedding,
                        "user_id": user_id,
                        "current_formatted_time": current_formatted_time,
                        "source_person_uid": source_person_uid_to_use
                    }
                else:
                    # Non-person or existing person - standard MERGE on (name, user_id)
                    on_create_clauses = [
                        "source.created_at = $current_formatted_time",
                        "source.embedding = $source_embedding"
                    ]
                    
                    cypher = f"""
                        MATCH (destination)
                        WHERE elementId(destination) = $destination_id
                        MERGE (source:{source_type} {{name: $source_name, user_id: $user_id}})
                        ON CREATE SET
                            {", ".join(on_create_clauses)}
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
                        "destination_id": destination_node_search_result[0]["elementId(destination_candidate)"],
                        "source_name": source,
                        "relationship": relationship,
                        "source_type": source_type,
                        "source_embedding": source_embedding,
                        "user_id": user_id,
                        "current_formatted_time": current_formatted_time,
                    }
                params.update(rel_params)

                logger.debug(f"Executing Cypher (destination exists): {cypher} with params: {params}")
                resp = self.graph.query(cypher, params=params)
                logger.debug(f"Graph query response: {resp}")
                results.append(resp)

            elif source_node_search_result and destination_node_search_result:
                current_formatted_time = datetime.now(pytz.utc).strftime('%Y-%m-%d %H:%M:%S')
                additional_set_properties_str, rel_params = _build_relationship_properties("r")

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
                params.update(rel_params)

                logger.debug(f"Executing Cypher (both exist): {cypher} with params: {params}")
                resp = self.graph.query(cypher, params=params)
                logger.debug(f"Graph query response: {resp}")
                results.append(resp)

            elif not source_node_search_result and not destination_node_search_result:
                current_formatted_time = datetime.now(pytz.utc).strftime('%Y-%m-%d %H:%M:%S')
                additional_set_properties_str, rel_params = _build_relationship_properties("rel")

                # For person nodes with new person_uid, include it in MERGE to ensure distinct nodes
                # For other nodes, use standard MERGE on (name, user_id)
                
                # Determine MERGE patterns for source
                if source_type == "person" and source_person_uid_to_use:
                    source_merge_pattern = f"{{name: $source_name, user_id: $user_id, person_uid: $source_person_uid}}"
                    source_on_create = ["n.created_at = $current_formatted_time", "n.embedding = $source_embedding"]
                else:
                    source_merge_pattern = f"{{name: $source_name, user_id: $user_id}}"
                    source_on_create = ["n.created_at = $current_formatted_time", "n.embedding = $source_embedding"]
                
                # Determine MERGE patterns for destination
                if destination_type == "person" and dest_person_uid_to_use:
                    dest_merge_pattern = f"{{name: $dest_name, user_id: $user_id, person_uid: $dest_person_uid}}"
                    dest_on_create = ["m.created_at = $current_formatted_time", "m.embedding = $dest_embedding"]
                else:
                    dest_merge_pattern = f"{{name: $dest_name, user_id: $user_id}}"
                    dest_on_create = ["m.created_at = $current_formatted_time", "m.embedding = $dest_embedding"]
                
                cypher = f"""
                    MERGE (n:{source_type} {source_merge_pattern})
                    ON CREATE SET 
                        {", ".join(source_on_create)}
                    ON MATCH SET 
                        n.embedding = $source_embedding
                    MERGE (m:{destination_type} {dest_merge_pattern})
                    ON CREATE SET 
                        {", ".join(dest_on_create)}
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
                
                if source_type == "person" and source_person_uid_to_use:
                    params["source_person_uid"] = source_person_uid_to_use
                if destination_type == "person" and dest_person_uid_to_use:
                    params["dest_person_uid"] = dest_person_uid_to_use
                params.update(rel_params)

                logger.debug(f"Executing Cypher (neither exist): {cypher} with params: {params}")
                resp = self.graph.query(cypher, params=params)
                logger.debug(f"Graph query response: {resp}")
                results.append(resp)
        
        logger.debug(f"Finished adding entities. Results: {results}")
        return results

    def _lookup_person_uid_by_name(self, person_name, user_id):
        """
        Look up an existing person's UID by normalized name for ownership attribution.
        """
        if not person_name or not user_id:
            return None

        query = """
        MATCH (p:person {name: $person_name, user_id: $user_id})
        RETURN p.person_uid AS person_uid
        LIMIT 1
        """
        params = {"person_name": person_name, "user_id": user_id}
        try:
            records = self.graph.query(query, params=params)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception(
                "Failed to look up person_uid for owner '%s': %s",
                person_name,
                exc,
            )
            return None

        if records:
            return records[0].get("person_uid")
        return None

    def _count_person_nodes(self, person_name, user_id):
        """
        Count how many person nodes exist for a given name and user.
        """
        if not person_name or not user_id:
            return 0

        query = """
        MATCH (p:person {name: $person_name, user_id: $user_id})
        RETURN count(p) AS person_count
        """
        params = {"person_name": person_name, "user_id": user_id}
        try:
            records = self.graph.query(query, params=params)
            if records:
                return records[0].get("person_count", 0) or 0
        except Exception as exc:
            logger.exception(
                "Failed to count person nodes for '%s': %s", person_name, exc
            )
        return 0

    def _fetch_single_person_profile(self, person_name, user_id):
        """
        Fetch a single person's profile (elementId, person_uid, name, facts) by exact name.

        Returns:
            dict or None: Profile matching _build_person_profiles format, or None if not found.
        """
        if not person_name or not user_id:
            return None

        query = """
        MATCH (p:person {name: $person_name, user_id: $user_id})
        WITH p, elementId(p) AS pid, p.person_uid AS person_uid
        OPTIONAL MATCH (p)-[r]->(m {user_id: $user_id})
        WITH p, pid, person_uid, collect({
            direction: 'out',
            relationship: type(r),
            target_name: m.name,
            target_uid: m.person_uid,
            target_type: coalesce(head([lab IN labels(m) WHERE lab <> 'person']), head(labels(m))),
            metadata: {
                weight: r.weight,
                is_uncertain: r.is_uncertain,
                status: r.status,
                emotion: r.emotion,
                start_date: r.start_date,
                end_date: r.end_date
            }
        }) AS outgoing
        OPTIONAL MATCH (m2 {user_id: $user_id})-[r2]->(p)
        WITH p, pid, person_uid, outgoing, collect({
            direction: 'in',
            relationship: type(r2),
            target_name: m2.name,
            target_uid: m2.person_uid,
            target_type: coalesce(head([lab IN labels(m2) WHERE lab <> 'person']), head(labels(m2))),
            metadata: {
                weight: r2.weight,
                is_uncertain: r2.is_uncertain,
                status: r2.status,
                emotion: r2.emotion,
                start_date: r2.start_date,
                end_date: r2.end_date
            }
        }) AS incoming
        RETURN pid AS element_id, person_uid, p.name AS name, outgoing + incoming AS facts
        LIMIT 1
        """
        params = {"person_name": person_name, "user_id": user_id}
        try:
            records = self.graph.query(query, params=params)
            if not records:
                return None
            row = records[0]
            return {
                "person_uid": row.get("person_uid"),
                "element_id": row.get("element_id"),
                "name": row.get("name"),
                "facts": row.get("facts") or [],
            }
        except Exception as exc:
            logger.exception(
                "Failed to fetch person profile for '%s': %s", person_name, exc
            )
            return None

    def _score_entity_candidates(self, entity_record, candidates, metadata=None):
        """
        Score candidate nodes for a given entity using cosine similarity, label matches, and context overlaps.

        Args:
            entity_record (dict): Contains at least {"name": str, "type": str}.
            candidates (list): List of candidate dicts with "node_id", "node_name", "node_labels", "similarity".
            metadata (dict): Optional context including:
                - context_nodes: iterable of related node names from the utterance
                - candidate_neighbors: dict mapping node_id -> iterable of neighbor names
                - profile_comparisons: dict mapping node_id -> comparison result ("match"/"contradict")
                - tie_delta: float threshold for logging near-ties

        Returns:
            dict with keys: candidate (best match or None), score (float), tie (bool), scores (list of per-candidate scores)
        """
        if not candidates:
            return {"candidate": None, "score": 0.0, "tie": False, "scores": []}

        metadata = metadata or {}
        tie_delta = metadata.get("tie_delta", self.CANDIDATE_TIE_DELTA)
        profile_comparisons = metadata.get("profile_comparisons") or {}

        def _normalize_nodes(raw_nodes, default_label=None):
            normalized = []
            for entry in raw_nodes or []:
                if entry is None:
                    continue
                if isinstance(entry, dict):
                    name = entry.get("name") or entry.get("target_name")
                    labels = entry.get("labels") or entry.get("label")
                    if not labels:
                        labels = entry.get("target_label") or entry.get("target_type")
                    embedding = entry.get("embedding") or entry.get("target_embedding")
                    descriptor = entry.get("descriptor")
                else:
                    name = str(entry)
                    labels = default_label
                    embedding = None
                    descriptor = None
                labels_list = []
                if isinstance(labels, (list, tuple, set)):
                    labels_list = [str(label).lower() for label in labels if label]
                elif labels:
                    labels_list = [str(labels).lower()]
                primary_label = labels_list[0] if labels_list else (default_label or "unknown")
                normalized.append(
                    {
                        "name": name,
                        "name_normalized": (name or "").lower(),
                        "labels": labels_list,
                        "primary_label": (primary_label or "unknown").lower(),
                        "embedding": embedding,
                        "descriptor": descriptor,
                    }
                )
            return normalized

        def _group_by_label(nodes):
            grouped = {}
            for node in nodes:
                label = node.get("primary_label") or "unknown"
                grouped.setdefault(label, []).append(node)
            return grouped

        def _resolve_embedding(node_entry):
            embedding = node_entry.get("embedding")
            if embedding:
                return embedding
            descriptor = node_entry.get("descriptor") or {}
            return descriptor.get("embedding")

        def _bucket_overlap(context_nodes, neighbor_nodes):
            match_count = 0
            best_similarity = 0.0
            for ctx in context_nodes:
                ctx_embedding = ctx.get("embedding")
                if not ctx_embedding:
                    continue
                for neighbor in neighbor_nodes:
                    neighbor_embedding = _resolve_embedding(neighbor)
                    if not neighbor_embedding:
                        continue
                    similarity = self._cosine_similarity(ctx_embedding, neighbor_embedding)
                    if similarity > best_similarity:
                        best_similarity = similarity
                    if similarity >= self.CONTEXT_MATCH_THRESHOLD:
                        match_count += 1
            return match_count, best_similarity

        context_nodes = _normalize_nodes(
            metadata.get("context_nodes"),
            default_label=(entity_record.get("type") or "unknown").lower(),
        )
        context_buckets = _group_by_label(context_nodes)
        effective_context_labels = {
            label
            for label, nodes in context_buckets.items()
            if any(node.get("embedding") for node in nodes)
        }
        context_bucket_count = len(effective_context_labels)

        candidate_neighbor_buckets = {}
        for node_id, neighbor_entries in (metadata.get("candidate_neighbors") or {}).items():
            normalized_neighbors = _normalize_nodes(neighbor_entries)
            candidate_neighbor_buckets[node_id] = _group_by_label(normalized_neighbors)

        entity_name = (entity_record.get("name") or "").lower()
        entity_type = (entity_record.get("type") or "unknown").lower()

        scored_entries = []
        for candidate in candidates:
            similarity = candidate.get("similarity") or 0.0
            score = similarity
            candidate_name = (candidate.get("node_name") or "").lower()
            if candidate_name == entity_name:
                score += 0.05

            candidate_labels = [
                label.lower() for label in (candidate.get("node_labels") or []) if label
            ]
            if entity_type and entity_type in candidate_labels:
                score += 0.03

            overlap_labels = set()
            missing_labels = set()
            penalty_log = []
            neighbor_buckets = candidate_neighbor_buckets.get(
                candidate.get("node_id"), {}
            )

            for label in effective_context_labels:
                label_context_nodes = context_buckets.get(label, [])
                label_neighbor_nodes = neighbor_buckets.get(label, [])
                match_count, _ = _bucket_overlap(
                    label_context_nodes,
                    label_neighbor_nodes,
                )
                if match_count > 0:
                    overlap_labels.add(label)
                    bucket_bonus = min(0.12, self.CONTEXT_MATCH_BONUS * match_count)
                    score += bucket_bonus
                else:
                    missing_labels.add(label)
                    penalty = self.CONTEXT_BUCKET_PENALTY
                    score -= penalty
                    penalty_log.append(
                        {
                            "label": label,
                            "penalty": penalty,
                            "reason": "no_overlap",
                        }
                    )
                    logger.info(
                        "[context_mismatch] entity=%s candidate=%s label=%s penalty=%.3f",
                        entity_record.get("name"),
                        candidate.get("node_name"),
                        label,
                        penalty,
                    )

            profile_result = profile_comparisons.get(candidate.get("node_id"))
            if profile_result == "match":
                score += 0.15
            elif profile_result == "contradict":
                score -= self.PROFILE_CONTRADICTION_PENALTY
                penalty_log.append(
                    {
                        "type": "profile_contradiction",
                        "penalty": self.PROFILE_CONTRADICTION_PENALTY,
                    }
                )
                logger.info(
                    "[candidate_penalized] entity=%s candidate=%s reason=profile_contradiction penalty=%.3f",
                    entity_record.get("name"),
                    candidate.get("node_name"),
                    self.PROFILE_CONTRADICTION_PENALTY,
                )

            if context_bucket_count and not overlap_labels:
                prior_score = score
                score *= self.NO_CONTEXT_OVERLAP_SCALE
                penalty_log.append(
                    {
                        "type": "no_context_overlap",
                        "scale": self.NO_CONTEXT_OVERLAP_SCALE,
                        "before": round(prior_score, 4),
                        "after": round(score, 4),
                    }
                )
                logger.info(
                    "[candidate_penalized] entity=%s candidate=%s reason=no_context_overlap before=%.4f after=%.4f",
                    entity_record.get("name"),
                    candidate.get("node_name"),
                    prior_score,
                    score,
                )

            scored_entries.append(
                (
                    score,
                    candidate,
                    {
                        "node_id": candidate.get("node_id"),
                        "name": candidate.get("node_name"),
                        "score": round(score, 4),
                        "similarity": round(similarity, 4),
                        "overlap_labels": sorted(overlap_labels),
                        "missing_labels": sorted(missing_labels),
                        "penalties": [dict(p) for p in penalty_log],
                        "penalties_applied": bool(penalty_log),
                        "profile_result": profile_result,
                    },
                )
            )

        scored_entries.sort(key=lambda item: item[0], reverse=True)
        best_score, best_candidate, best_details = scored_entries[0]
        tie = False
        if len(scored_entries) > 1:
            second_score = scored_entries[1][0]
            delta = abs(best_score - second_score)
            if delta <= tie_delta:
                tie = True
                logger.info(
                    "[candidate_tie] entity=%s score1=%.4f score2=%.4f delta=%.4f",
                    entity_record.get("name"),
                    best_score,
                    second_score,
                    delta,
                )

        return {
            "candidate": best_candidate,
            "score": best_score,
            "tie": tie,
            "scores": [entry for _, _, entry in scored_entries],
            "overlap_labels": best_details.get("overlap_labels", []),
            "penalties_applied": best_details.get("penalties_applied", False),
            "context_bucket_count": context_bucket_count,
        }

    def _get_candidate_neighbor_names(self, candidate_ids, user_id):
        """
        Fetch neighbor names for candidate nodes to provide relationship context for scoring.
        """
        if not candidate_ids:
            return {}

        cypher = """
        UNWIND $candidate_ids AS candidate_id
        MATCH (n)
        WHERE elementId(n) = candidate_id AND n.user_id = $user_id
        OPTIONAL MATCH (n)-[r]-(neighbor {user_id: $user_id})
        WITH candidate_id, collect(DISTINCT neighbor)[..$neighbor_limit] AS neighbors
        RETURN candidate_id AS node_id,
               [neighbor IN neighbors WHERE neighbor IS NOT NULL |
                    {
                        name: neighbor.name,
                        labels: labels(neighbor),
                        embedding: neighbor.embedding,
                        descriptor: head([
                            (neighbor)-[descriptor_rel]-(descriptor {user_id: $user_id})
                            WHERE descriptor.embedding IS NOT NULL
                              AND NOT 'person' IN labels(descriptor)
                              AND descriptor <> neighbor
                            | {
                                name: descriptor.name,
                                labels: labels(descriptor),
                                embedding: descriptor.embedding,
                                relationship: toLower(type(descriptor_rel))
                            }
                        ])
                    }
               ] AS neighbor_details
        """
        params = {
            "candidate_ids": candidate_ids,
            "user_id": user_id,
            "neighbor_limit": self.CANDIDATE_NEIGHBOR_LIMIT,
        }
        records = self.graph.query(cypher, params=params)
        neighbor_map = {}
        for record in records:
            node_id = record.get("node_id")
            neighbor_details = []
            for neighbor in record.get("neighbor_details") or []:
                descriptor = neighbor.get("descriptor")
                descriptor_info = None
                if descriptor:
                    descriptor_info = {
                        "name": descriptor.get("name"),
                        "labels": [
                            str(label).lower()
                            for label in (descriptor.get("labels") or [])
                            if label
                        ],
                        "embedding": descriptor.get("embedding"),
                        "relationship": descriptor.get("relationship"),
                    }
                neighbor_details.append(
                    {
                        "name": neighbor.get("name"),
                        "labels": [
                            str(label).lower()
                            for label in (neighbor.get("labels") or [])
                            if label
                        ],
                        "embedding": neighbor.get("embedding"),
                        "descriptor": descriptor_info,
                    }
                )
            neighbor_map[node_id] = neighbor_details
        return neighbor_map

    def _remove_spaces_from_entities(self, entity_list):
        """
        Process entities by:
        1. Converting entity names to lowercase and replacing spaces with underscores
        2. Ensuring all required parameters are present with default values if missing
        3. Filter out any literal pronoun nodes that slipped through
        """
        # Pronouns that should never become nodes
        filtered_entities = []
        for item in entity_list:
            item["source"] = item["source"].lower().replace(" ", "_")
            item["relationship"] = item["relationship"].lower().replace(" ", "_")
            item["destination"] = item["destination"].lower().replace(" ", "_")
            owner_person_name = item.get("owner_person_name")
            if isinstance(owner_person_name, str):
                owner_person_name = owner_person_name.strip()
                if owner_person_name:
                    owner_person_name = owner_person_name.lower().replace(" ", "_")
                else:
                    owner_person_name = None
            else:
                owner_person_name = None
            item["owner_person_name"] = owner_person_name

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

    def _search_source_node(self, source_embedding, user_id, label=None, limit=None):
        return self._search_node_candidates(
            embedding=source_embedding,
            user_id=user_id,
            label=label,
            candidate_alias="source_candidate",
            limit=limit,
        )

    def _search_destination_node(self, destination_embedding, user_id, label=None, limit=None):
        return self._search_node_candidates(
            embedding=destination_embedding,
            user_id=user_id,
            label=label,
            candidate_alias="destination_candidate",
            limit=limit,
        )

    def _search_node_candidates(self, embedding, user_id, label, candidate_alias, limit=None):
        """
        Run a label-aware cosine similarity search for potential node re-use.
        """
        limit = limit or self.NODE_SEARCH_CANDIDATE_LIMIT
        label_filter = label.lower() if isinstance(label, str) else None
        if label_filter in (None, "", "unknown"):
            label_filter = None

        candidates = []
        label_threshold = self._get_label_threshold(label_filter)
        if label_filter:
            candidates = self._run_node_similarity_query(
                embedding=embedding,
                user_id=user_id,
                threshold=label_threshold,
                label_filter=label_filter,
                require_label_match=True,
                candidate_alias=candidate_alias,
                limit=limit,
            )
            if not candidates:
                logger.info(
                    "[node_label_fallback] alias=%s label=%s threshold=%.3f",
                    candidate_alias,
                    label_filter,
                    label_threshold,
                )

        if not candidates:
            candidates = self._run_node_similarity_query(
                embedding=embedding,
                user_id=user_id,
                threshold=self._get_label_threshold("default"),
                label_filter=None,
                require_label_match=False,
                candidate_alias=candidate_alias,
                limit=limit,
            )
        return candidates

    def _run_node_similarity_query(
        self,
        embedding,
        user_id,
        threshold,
        label_filter,
        require_label_match,
        candidate_alias,
        limit,
    ):
        cypher = f"""
            MATCH ({candidate_alias})
            WHERE {candidate_alias}.embedding IS NOT NULL 
              AND {candidate_alias}.user_id = $user_id
            WITH {candidate_alias}, [label IN labels({candidate_alias}) | toLower(label)] AS node_labels,
                 round(
                    reduce(dot = 0.0, i IN range(0, size({candidate_alias}.embedding)-1) |
                        dot + {candidate_alias}.embedding[i] * $node_embedding[i]) /
                    (sqrt(reduce(l2 = 0.0, i IN range(0, size({candidate_alias}.embedding)-1) |
                        l2 + {candidate_alias}.embedding[i] * {candidate_alias}.embedding[i])) *
                    sqrt(reduce(l2 = 0.0, i IN range(0, size($node_embedding)-1) |
                        l2 + $node_embedding[i] * $node_embedding[i])))
                , 4) AS similarity
            WHERE similarity >= $threshold
              AND (
                $require_label_match = false OR (
                    $label_filter IS NOT NULL AND $label_filter IN node_labels
                )
              )
            WITH {candidate_alias}, node_labels, similarity
            ORDER BY similarity DESC
            LIMIT $limit
            RETURN elementId({candidate_alias}) AS node_id,
                   {candidate_alias}.name AS node_name,
                   node_labels AS node_labels,
                   {candidate_alias}.person_uid AS person_uid,
                   similarity
        """
        params = {
            "node_embedding": embedding,
            "user_id": user_id,
            "threshold": threshold,
            "label_filter": label_filter,
            "require_label_match": require_label_match,
            "limit": limit,
        }
        records = self.graph.query(cypher, params=params)
        return [
            {
                "node_id": record.get("node_id"),
                "node_name": record.get("node_name"),
                "node_labels": record.get("node_labels") or [],
                "person_uid": record.get("person_uid"),
                "similarity": record.get("similarity") or 0.0,
            }
            for record in records
        ]

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

    def reset(self):
        """Reset the graph by clearing all nodes and relationships."""
        logger.warning("Clearing graph...")
        cypher_query = """
        MATCH (n) DETACH DELETE n
        """
        return self.graph.query(cypher_query)
    
    def _build_person_profiles(self, search_output, entity_type_map):
        """
        Build profiles for each person node found in the search output.
        
        FLATTENED STRUCTURE: All relationships are stored as normalized fact records
        in a single list, making comparison more robust and preventing information loss.
        
        Args:
            search_output: List of dictionaries from _search_graph_db, each containing:
                - source, source_id, source_labels, source_person_uid
                - destination, destination_id, destination_labels, destination_person_uid
                - relatationship, relation_id, weight, status, etc.
            entity_type_map: Dict mapping entity names to their types from extraction
                
        Returns:
            Dict keyed by normalized person name -> list of profile dicts.
            Each profile dict contains:
                - person_uid: str or None
                - element_id: Neo4j element ID
                - name: normalized name
                - facts: list of normalized fact records, each containing:
                    * direction: "out" or "in"
                    * relationship: relationship type
                    * target_name: normalized name of the other node
                    * target_uid: person_uid if target is a person, else None
                    * target_type: inferred type (from labels or entity_type_map)
                    * metadata: dict with weight, status, emotion, etc. (excludes volatile timestamps)
        """
        profiles = {}
        
        if not search_output:
            return profiles
        
        for row in search_output:
            # Process source if it's a person
            source_name = row.get("source")
            source_labels = row.get("source_labels", [])
            source_type = entity_type_map.get(source_name, "unknown")
            
            # Check if source is a person (either from labels or entity_type_map)
            is_source_person = (
                source_type == "person" or
                "person" in [label.lower() for label in source_labels] if source_labels else False
            )
            
            if is_source_person and source_name:
                if source_name not in profiles:
                    profiles[source_name] = []
                
                # Find or create profile for this specific node instance
                source_id = row.get("source_id")
                source_person_uid = row.get("source_person_uid")
                
                # Check if we already have a profile for this specific node (by element_id)
                existing_profile = None
                for prof in profiles[source_name]:
                    if prof["element_id"] == source_id:
                        existing_profile = prof
                        break
                
                if not existing_profile:
                    existing_profile = {
                        "person_uid": source_person_uid,
                        "element_id": source_id,
                        "name": source_name,
                        "facts": []  # Flattened list of fact records
                    }
                    profiles[source_name].append(existing_profile)
                
                # Build outgoing fact record
                dest_name = row.get("destination")
                dest_labels = row.get("destination_labels", [])
                dest_person_uid = row.get("destination_person_uid")
                dest_type = entity_type_map.get(dest_name, "unknown")
                
                # Infer target type from labels if available
                if dest_labels:
                    # Use the first label that's not a generic one
                    for label in dest_labels:
                        label_lower = label.lower()
                        if label_lower in ["person", "role", "organization", "location", "concept", "activity"]:
                            dest_type = label_lower
                            break
                
                fact_record = {
                    "direction": "out",
                    "relationship": row.get("relatationship"),
                    "target_name": dest_name,
                    "target_uid": dest_person_uid,  # Will be None for non-person nodes
                    "target_type": dest_type,
                    "metadata": {
                        "weight": row.get("weight"),
                        "is_uncertain": row.get("is_uncertain"),
                        "status": row.get("status"),
                        "emotion": row.get("emotion"),
                        # Exclude volatile timestamps: last_mentioned, usage_count
                        # Include dates that represent relationship timing:
                        "start_date": row.get("start_date"),
                        "end_date": row.get("end_date"),
                    }
                }
                existing_profile["facts"].append(fact_record)
            
            # Process destination if it's a person
            dest_name = row.get("destination")
            dest_labels = row.get("destination_labels", [])
            dest_type = entity_type_map.get(dest_name, "unknown")
            
            is_dest_person = (
                dest_type == "person" or
                "person" in [label.lower() for label in dest_labels] if dest_labels else False
            )
            
            if is_dest_person and dest_name:
                if dest_name not in profiles:
                    profiles[dest_name] = []
                
                dest_id = row.get("destination_id")
                dest_person_uid = row.get("destination_person_uid")
                
                # Find or create profile
                existing_profile = None
                for prof in profiles[dest_name]:
                    if prof["element_id"] == dest_id:
                        existing_profile = prof
                        break
                
                if not existing_profile:
                    existing_profile = {
                        "person_uid": dest_person_uid,
                        "element_id": dest_id,
                        "name": dest_name,
                        "facts": []
                    }
                    profiles[dest_name].append(existing_profile)
                
                # Build incoming fact record
                source_name_for_incoming = row.get("source")
                source_labels_for_incoming = row.get("source_labels", [])
                source_person_uid_for_incoming = row.get("source_person_uid")
                source_type_for_incoming = entity_type_map.get(source_name_for_incoming, "unknown")
                
                # Infer source type from labels
                if source_labels_for_incoming:
                    for label in source_labels_for_incoming:
                        label_lower = label.lower()
                        if label_lower in ["person", "role", "organization", "location", "concept", "activity"]:
                            source_type_for_incoming = label_lower
                            break
                
                fact_record = {
                    "direction": "in",
                    "relationship": row.get("relatationship"),
                    "target_name": source_name_for_incoming,
                    "target_uid": source_person_uid_for_incoming,
                    "target_type": source_type_for_incoming,
                    "metadata": {
                        "weight": row.get("weight"),
                        "is_uncertain": row.get("is_uncertain"),
                        "status": row.get("status"),
                        "emotion": row.get("emotion"),
                        "start_date": row.get("start_date"),
                        "end_date": row.get("end_date"),
                    }
                }
                existing_profile["facts"].append(fact_record)
        
        logger.debug(f"Built person profiles: {profiles}")
        return profiles
    
    def _compare_person_profiles(self, existing_profile, new_profile):
        """
        Compare an existing person profile with a new one using signature-based matching.
        
        SIGNATURE-BASED MATCHING: Facts are compared using (direction, relationship, target_name) 
        as the primary signature, optionally enhanced with target_uid for person nodes.
        
        Args:
            existing_profile: Dict with person_uid, element_id, name, and facts list
            new_profile: Dict with the same structure (but person_uid will be None)
                
        Returns:
            str: One of "match", "contradict", or "no_overlap"
                - "match": Strong evidence they're the same person
                - "contradict": Evidence they're different people
                - "no_overlap": Not enough information to determine
        """
        existing_facts = existing_profile["facts"]
        new_facts = new_profile["facts"]
        
        # Build comparable signatures for existing facts
        # Signature = (direction, relationship, target_name, target_uid if present)
        # Only include target_uid in signature if it's actually populated
        # This allows exact name matches to work when UIDs aren't available
        existing_signatures = set()
        for fact in existing_facts:
            # Create signature tuple, excluding volatile fields
            target_uid = fact.get("target_uid")
            if target_uid:
                sig = (
                    fact["direction"],
                    fact["relationship"],
                    fact["target_name"],
                    target_uid
                )
            else:
                # No UID available, use 3-tuple signature (name-based matching)
                sig = (
                    fact["direction"],
                    fact["relationship"],
                    fact["target_name"]
                )
            existing_signatures.add(sig)
        
        # Build comparable signatures for new facts
        new_signatures = set()
        for fact in new_facts:
            target_uid = fact.get("target_uid")
            if target_uid:
                sig = (
                    fact["direction"],
                    fact["relationship"],
                    fact["target_name"],
                    target_uid
                )
            else:
                sig = (
                    fact["direction"],
                    fact["relationship"],
                    fact["target_name"]
                )
            new_signatures.add(sig)
        
        # Calculate exact matches on signatures
        exact_matches = existing_signatures.intersection(new_signatures)
        
        # Score for matching evidence
        match_score = 0
        contradict_score = 0
        
        # Exact signature matches are the primary signal
        if exact_matches:
            match_count = len(exact_matches)
            if match_count >= 3:
                match_score += 15  # Multiple exact matches = very strong signal
            elif match_count == 2:
                match_score += 10  # Two matches = strong signal
            else:
                match_score += 6   # One match = medium signal
            logger.debug(f"Exact signature matches ({match_count}): {exact_matches}")
        
        # Check for contradictions: same relationship type but different target
        # This indicates mutually exclusive facts
        existing_rel_map = {}  # (direction, relationship) -> set of target_names
        for fact in existing_facts:
            key = (fact["direction"], fact["relationship"])
            if key not in existing_rel_map:
                existing_rel_map[key] = set()
            existing_rel_map[key].add((fact["target_name"], fact.get("target_uid")))
        
        new_rel_map = {}
        for fact in new_facts:
            key = (fact["direction"], fact["relationship"])
            if key not in new_rel_map:
                new_rel_map[key] = set()
            new_rel_map[key].add((fact["target_name"], fact.get("target_uid")))
        
        # Define single-valued outward relationships that are typically exclusive
        # These are relationships where a person usually has only one target at a time
        EXCLUSIVE_OUT_RELATIONSHIPS = {
            "works_at", "employed_by", "lives_in", "resides_in", "works_in",
            "attends", "studies_at", "enrolled_in", "married_to", "spouse_of",
            "reports_to", "manages", "ceo_of", "founder_of", "owns_company"
        }
        
        # Look for contradictory facts
        for rel_key in new_rel_map:
            if rel_key in existing_rel_map:
                existing_targets = existing_rel_map[rel_key]
                new_targets = new_rel_map[rel_key]
                
                # If they have no overlap, it might be a contradiction
                if not existing_targets.intersection(new_targets):
                    direction, relationship = rel_key
                    
                    # Check for contradictions based on relationship type and direction
                    
                    # 1. Different person_uids for same relationship = different people
                    if any(uid1 and uid2 and uid1 != uid2 
                           for (name1, uid1) in existing_targets 
                           for (name2, uid2) in new_targets):
                        # Same relationship points to different person_uids
                        contradict_score += 5
                        logger.debug(f"Contradiction: Same relationship points to different person UIDs")
                    
                    # 2. Single-valued outward relationships to non-person nodes
                    # (e.g., works_at, lives_in) are typically exclusive
                    elif direction == "out" and relationship in EXCLUSIVE_OUT_RELATIONSHIPS:
                        # Check if targets are non-person nodes (no UIDs)
                        # If all targets lack UIDs, they're likely non-person nodes (jobs, locations, etc.)
                        all_existing_no_uid = all(uid is None for (name, uid) in existing_targets)
                        all_new_no_uid = all(uid is None for (name, uid) in new_targets)
                        
                        if all_existing_no_uid and all_new_no_uid:
                            # Different non-person targets for exclusive relationship = contradiction
                            contradict_score += 6
                            logger.debug(f"Contradiction: Different exclusive '{relationship}' targets: existing={existing_targets}, new={new_targets}")
        
        # Decision logic
        logger.debug(f"Profile comparison scores - match: {match_score}, contradict: {contradict_score}")
        
        # If we have strong contradictory evidence, return contradict
        if contradict_score >= 5:
            return "contradict"
        
        # If we have strong matching evidence, return match
        if match_score >= 8:
            return "match"
        
        # If we have some matching evidence but not strong, still consider it a match
        if match_score >= 3 and contradict_score == 0:
            return "match"
        
        # Otherwise, not enough overlap to determine
        return "no_overlap"
    
    def _select_person_candidate(self, person_name, new_profile, existing_profiles):
        """
        Select which existing person node (if any) to reuse for a new mention.
        
        Args:
            person_name: str, the normalized name of the person
            new_profile: Dict with the new person's relationship profile
            existing_profiles: List of existing profile dicts for this name
                
        Returns:
            Dict with keys:
                - decision: str, one of "reuse", "new", "ambiguous", "unconfirmed"
                - element_id: str or None, the Neo4j element ID to reuse (if decision="reuse")
                - person_uid: str or None, the person_uid to reuse (if decision="reuse")
                - matched_profile: dict or None, the full profile that matched
                - all_comparisons: list of tuples (profile, comparison_result) for debugging
        """
        if not existing_profiles:
            return {
                "decision": "new",
                "element_id": None,
                "person_uid": None,
                "matched_profile": None,
                "all_comparisons": []
            }
        
        # Compare new profile against each existing profile
        comparisons = []
        matches = []
        contradictions = []
        no_overlaps = []
        
        for existing_profile in existing_profiles:
            result = self._compare_person_profiles(existing_profile, new_profile)
            comparisons.append((existing_profile, result))
            
            if result == "match":
                matches.append(existing_profile)
            elif result == "contradict":
                contradictions.append(existing_profile)
            else:  # no_overlap
                no_overlaps.append(existing_profile)
        
        logger.debug(f"Person candidate selection for '{person_name}': {len(matches)} matches, {len(contradictions)} contradictions, {len(no_overlaps)} no overlaps")
        
        # If there's exactly one existing profile and no contradictions, reuse it even without overlapping facts.
        if len(existing_profiles) == 1 and not contradictions:
            primary_profile = existing_profiles[0]
            return {
                "decision": "reuse",
                "element_id": primary_profile["element_id"],
                "person_uid": primary_profile["person_uid"],
                "matched_profile": primary_profile if matches else None,
                "all_comparisons": comparisons
            }
        
        # Decision logic
        if len(matches) == 1:
            # Clear single match - reuse this node
            matched = matches[0]
            return {
                "decision": "reuse",
                "element_id": matched["element_id"],
                "person_uid": matched["person_uid"],
                "matched_profile": matched,
                "all_comparisons": comparisons
            }
        
        elif len(matches) > 1:
            # Multiple matches - ambiguous
            # This shouldn't happen often, but could if two existing nodes have very similar profiles
            logger.warning(f"Ambiguous person match for '{person_name}': {len(matches)} existing nodes matched")
            return {
                "decision": "ambiguous",
                "element_id": None,
                "person_uid": None,
                "matched_profile": None,
                "all_comparisons": comparisons,
                "candidates": matches
            }
        
        elif len(contradictions) > 0 and len(no_overlaps) > 0:
            # Some contradictions, some no overlaps - suggests there might be multiple people
            # but we're not sure which (if any) the new one corresponds to
            logger.info(f"Mixed signals for '{person_name}': {len(contradictions)} contradictions, {len(no_overlaps)} unclear")
            return {
                "decision": "ambiguous",
                "element_id": None,
                "person_uid": None,
                "matched_profile": None,
                "all_comparisons": comparisons,
                "candidates": existing_profiles
            }
        
        elif len(contradictions) > 0:
            # All existing profiles contradict - likely a new person
            logger.info(f"All existing profiles for '{person_name}' contradict new data - creating new node")
            return {
                "decision": "new",
                "element_id": None,
                "person_uid": None,
                "matched_profile": None,
                "all_comparisons": comparisons
            }
        
        else:
            # Only no_overlaps - not enough information
            # Be conservative: ask for confirmation before creating a new node
            logger.info(f"No overlap with existing profiles for '{person_name}' - unconfirmed")
            return {
                "decision": "unconfirmed",
                "element_id": None,
                "person_uid": None,
                "matched_profile": None,
                "all_comparisons": comparisons,
                "existing_count": len(existing_profiles)
            }
    
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
