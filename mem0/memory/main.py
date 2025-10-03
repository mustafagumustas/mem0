import concurrent
import hashlib
import json
import logging
import uuid
import warnings
from datetime import datetime
from typing import Any, Dict

import pytz
from pydantic import ValidationError

from mem0.configs.base import MemoryConfig, MemoryItem
from mem0.configs.prompts import get_update_memory_messages
from mem0.memory.base import MemoryBase
from mem0.memory.setup import setup_config
from mem0.memory.storage import SQLiteManager
from mem0.memory.telemetry import capture_event
from mem0.memory.utils import (
    get_fact_retrieval_messages,
    parse_messages,
    parse_vision_messages,
    remove_code_blocks,
)
from mem0.utils.factory import EmbedderFactory, LlmFactory, VectorStoreFactory

# Setup user config
setup_config()

logger = logging.getLogger(__name__)


class Memory(MemoryBase):
    def __init__(self, config: MemoryConfig = MemoryConfig()):
        self.config = config

        self.custom_fact_extraction_prompt = self.config.custom_fact_extraction_prompt
        self.custom_update_memory_prompt = self.config.custom_update_memory_prompt
        self.embedding_model = EmbedderFactory.create(self.config.embedder.provider, self.config.embedder.config)
        self.vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, self.config.vector_store.config
        )
        self.llm = LlmFactory.create(self.config.llm.provider, self.config.llm.config)
        self.db = SQLiteManager(self.config.history_db_path)
        self.collection_name = self.config.vector_store.config.collection_name
        self.api_version = self.config.version

        self.enable_graph = False

        if self.config.graph_store.config:
            from mem0.memory.graph_memory import MemoryGraph

            self.graph = MemoryGraph(self.config)
            self.enable_graph = True

        capture_event("mem0.init", self)

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]):
        try:
            config = cls._process_config(config_dict)
            config = MemoryConfig(**config_dict)
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise
        return cls(config)

    @staticmethod
    def _process_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        if "graph_store" in config_dict:
            if "vector_store" not in config_dict and "embedder" in config_dict:
                config_dict["vector_store"] = {}
                config_dict["vector_store"]["config"] = {}
                config_dict["vector_store"]["config"]["embedding_model_dims"] = config_dict["embedder"]["config"][
                    "embedding_dims"
                ]
        try:
            return config_dict
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise

    def _extract_role_references(self, messages, filters):
        """
        Extract role references from messages using the LLM.
        
        Args:
            messages (list): List of message dicts
            filters (dict): Filters including user_id
            
        Returns:
            dict: Result with keys:
                - entities: List of explicit entities
                - references: List of role references needing resolution
        """
        from mem0.graphs.tools import (
            EXTRACT_ROLE_REFERENCES_TOOL,
            EXTRACT_ROLE_REFERENCES_STRUCT_TOOL,
        )
        from mem0.graphs.utils import EXTRACT_ROLE_REFERENCES_PROMPT
        
        # Aggregate only user/assistant messages (not system)
        content_to_analyze = "\n".join([
            msg["content"] for msg in messages 
            if msg.get("role") not in ["system"] and "content" in msg
        ])
        
        if not content_to_analyze.strip():
            return {"entities": [], "references": []}
        
        # Select tool based on LLM provider
        _tools = [EXTRACT_ROLE_REFERENCES_TOOL]
        if self.config.llm.provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [EXTRACT_ROLE_REFERENCES_STRUCT_TOOL]
        
        try:
            # Call LLM with role reference extraction tool
            response = self.llm.generate_response(
                messages=[
                    {
                        "role": "system",
                        "content": EXTRACT_ROLE_REFERENCES_PROMPT.replace("USER_ID", filters.get("user_id", "user"))
                    },
                    {
                        "role": "user",
                        "content": content_to_analyze
                    }
                ],
                tools=_tools
            )
            
            # Parse the tool call response
            if response and response.get("tool_calls"):
                tool_call = response["tool_calls"][0]
                if tool_call["name"] == "extract_role_references":
                    result = tool_call["arguments"]
                    logger.info(f"Extracted {len(result.get('entities', []))} entities and {len(result.get('references', []))} role references")
                    return result
            
            # No tool call or unexpected response
            logger.warning("No role references extracted from LLM response")
            return {"entities": [], "references": []}
            
        except Exception as e:
            logger.error(f"Error extracting role references: {e}")
            return {"entities": [], "references": []}

    def _rewrite_messages_with_resolved_names(self, messages, resolved_references):
        """
        Rewrite messages by replacing role references with their canonical resolved names.
        
        Args:
            messages (list): List of message dicts
            resolved_references (list): List of resolution result dicts with:
                - reference: The original reference dict
                - resolved_name: The canonical name to use
                
        Returns:
            list: New list of messages with references replaced
        """
        rewritten_messages = []
        
        for message in messages:
            if "content" not in message or message.get("role") == "system":
                rewritten_messages.append(message.copy())
                continue
            
            content = message["content"]
            
            # Replace each resolved reference with its canonical name
            for res_ref in resolved_references:
                if res_ref["status"] == "resolved" and "resolved_name" in res_ref:
                    reference = res_ref["reference"]
                    raw_text = reference["raw_text"]
                    resolved_name = res_ref["resolved_name"]
                    
                    # Replace the raw text span with the resolved name
                    # Case-insensitive replacement
                    import re
                    pattern = re.compile(re.escape(raw_text), re.IGNORECASE)
                    content = pattern.sub(resolved_name, content)
                    
                    logger.debug(f"Rewrote '{raw_text}' -> '{resolved_name}' in message")
            
            # Create new message with rewritten content
            rewritten_message = message.copy()
            rewritten_message["content"] = content
            rewritten_messages.append(rewritten_message)
        
        return rewritten_messages

    def _resolve_from_vector(self, reference, filters, limit=5):
        """
        Attempt to resolve a role reference using vector store search.
        
        Args:
            reference (dict): Reference dict with keys: raw_text, normalized_role, normalized_entity, side
            filters (dict): Search filters (user_id, agent_id, run_id)
            limit (int): Maximum number of results to search
            
        Returns:
            list: List of candidate dicts with keys: name, normalized_name, source
        """
        candidates = []
        
        # Search using the role phrase
        query = f"{reference['normalized_entity']} {reference['normalized_role']}"
        
        try:
            query_embedding = self.embedding_model.embed(query, "search")
            search_results = self.vector_store.search(
                query=query,
                vectors=query_embedding,
                limit=limit,
                filters=filters
            )
            
            # Extract potential names from the results
            import re
            for result in search_results:
                data = result.payload.get("data", "")
                
                # Simple heuristic: extract capitalized words that might be names
                # This is a basic approach - could be improved with NER
                words = re.findall(r'\b[A-Z][a-z]+\b', data)
                
                for word in words:
                    # Avoid common non-name words
                    if word.lower() not in ["i", "the", "a", "an", "this", "that", "user_id"]:
                        candidates.append({
                            "name": word,
                            "normalized_name": word.lower(),
                            "source": "vector",
                            "score": result.score if hasattr(result, 'score') else 0.0,
                            "context": data[:100]  # Store some context
                        })
            
            # Deduplicate by normalized name
            seen = {}
            unique_candidates = []
            for candidate in candidates:
                if candidate["normalized_name"] not in seen:
                    seen[candidate["normalized_name"]] = True
                    unique_candidates.append(candidate)
            
            logger.info(f"Found {len(unique_candidates)} vector candidates for '{reference['raw_text']}'")
            return unique_candidates
            
        except Exception as e:
            logger.error(f"Error in vector-based role resolution: {e}")
            return []

    def _render_clarification_question(self, role, candidates, side="source"):
        """
        Render a natural language clarification question.
        
        Args:
            role (str): The normalized role (e.g., "best_friend")
            candidates (list): List of candidate dicts with at least a "name" key
            side (str): "source" or "destination"
            
        Returns:
            str: A natural language question asking for clarification
        """
        # Convert role from snake_case to readable form
        readable_role = role.replace("_", " ")
        
        if len(candidates) == 0:
            # No candidates found - ask for new information
            return f"I don't have any information about your {readable_role}. Could you tell me who your {readable_role} is?"
        
        elif len(candidates) == 1:
            # Single candidate but we want confirmation
            return f"By '{readable_role}', do you mean {candidates[0]['name']}?"
        
        else:
            # Multiple candidates - ask which one
            names = [c["name"] for c in candidates]
            if len(names) == 2:
                names_str = f"{names[0]} or {names[1]}"
            else:
                names_str = ", ".join(names[:-1]) + f", or {names[-1]}"
            
            return f"I found multiple people who match '{readable_role}': {names_str}. Which {readable_role} were you referring to?"

    def _resolve_role_references(self, messages, references, filters):
        """
        Resolve role references against graph and vector store.
        
        Args:
            messages (list): List of message dicts
            references (list): List of reference dicts from role extraction
            filters (dict): Filters including user_id
            
        Returns:
            dict: Resolution result with structure:
                - status: "resolved", "clarification_needed", or "error"
                - resolved_references: List of successfully resolved references (if status="resolved")
                - clarification: Clarification payload (if status="clarification_needed")
        """
        if not references:
            return {"status": "resolved", "resolved_references": []}
        
        resolution_results = []
        
        for reference in references:
            # Skip if this doesn't need resolution (explicit entity already present)
            if reference.get("normalized_entity") not in ["USER_ID", "he", "she", "they", "it"]:
                # This is already a specific entity, not a role reference needing resolution
                resolution_results.append({
                    "reference": reference,
                    "status": "explicit",
                    "candidates": []
                })
                continue
            
            # Replace USER_ID with actual user_id for graph queries
            query_entity = filters["user_id"] if reference["normalized_entity"] == "USER_ID" else reference["normalized_entity"]
            
            graph_candidates = []
            if self.enable_graph:
                try:
                    graph_candidates = self.graph.get_role_candidates(
                        user_id=filters["user_id"],
                        normalized_role=reference["normalized_role"],
                        side=reference.get("side", "source")
                    )
                except Exception as e:
                    logger.error(f"Error getting graph candidates: {e}")
            
            # Optionally combine with vector evidence
            vector_candidates = self._resolve_from_vector(reference, filters, limit=5)
            
            # Merge candidates (prioritize graph, but include vector if graph is empty)
            all_candidates = graph_candidates.copy()
            
            # Add vector candidates that aren't already in graph candidates
            graph_names = {c["normalized_name"] for c in graph_candidates}
            for vc in vector_candidates:
                if vc["normalized_name"] not in graph_names:
                    all_candidates.append(vc)
            
            # Deduplicate and normalize
            unique_candidates = []
            seen_names = set()
            for candidate in all_candidates:
                norm_name = candidate["normalized_name"]
                if norm_name not in seen_names:
                    seen_names.add(norm_name)
                    unique_candidates.append(candidate)
            
            # Classify the result
            num_candidates = len(unique_candidates)
            
            if num_candidates == 0:
                # Zero candidates - need clarification
                resolution_results.append({
                    "reference": reference,
                    "status": "ambiguous",
                    "candidates": [],
                    "reason": "no_candidates"
                })
            elif num_candidates == 1:
                # Exactly one candidate - resolved!
                resolution_results.append({
                    "reference": reference,
                    "status": "resolved",
                    "candidates": unique_candidates,
                    "resolved_name": unique_candidates[0]["name"]
                })
            else:
                # Multiple candidates - need clarification
                resolution_results.append({
                    "reference": reference,
                    "status": "ambiguous",
                    "candidates": unique_candidates,
                    "reason": "multiple_candidates"
                })
        
        # Check if any references need clarification
        ambiguous_refs = [r for r in resolution_results if r["status"] == "ambiguous"]
        
        if ambiguous_refs:
            # Build clarification payload for the first ambiguous reference
            # (In a full implementation, you might handle multiple clarifications)
            first_ambiguous = ambiguous_refs[0]
            reference = first_ambiguous["reference"]
            candidates = first_ambiguous["candidates"]
            
            clarification_id = str(uuid.uuid4())
            message = self._render_clarification_question(
                role=reference["normalized_role"],
                candidates=candidates,
                side=reference.get("side", "source")
            )
            
            clarification_payload = {
                "clarification": {
                    "status": "pending",
                    "clarification_id": clarification_id,
                    "role": reference["normalized_role"],
                    "side": reference.get("side", "source"),
                    "trigger_span": reference["raw_text"],
                    "candidates": [{"name": c["name"], "source": c.get("source", "graph")} for c in candidates],
                    "message": message,
                    "context": {
                        "all_references": references,
                        "ambiguous_count": len(ambiguous_refs)
                    }
                }
            }
            
            # Log the clarification event
            logger.info(f"Role clarification needed: {reference['raw_text']} -> {len(candidates)} candidates")
            capture_event("mem0.role_clarification", self, {
                "role": reference["normalized_role"],
                "candidate_count": len(candidates),
                "reason": first_ambiguous["reason"]
            })
            
            return {
                "status": "clarification_needed",
                "clarification": clarification_payload
            }
        
        # All references resolved successfully
        return {
            "status": "resolved",
            "resolved_references": resolution_results
        }

    def add(
        self,
        messages,
        user_id=None,
        agent_id=None,
        run_id=None,
        metadata=None,
        filters=None,
        infer=True,
        prompt=None,
    ):
        """
        Create a new memory.

        Args:
            messages (str or List[Dict[str, str]]): Messages to store in the memory.
            user_id (str, optional): ID of the user creating the memory. Defaults to None.
            agent_id (str, optional): ID of the agent creating the memory. Defaults to None.
            run_id (str, optional): ID of the run creating the memory. Defaults to None.
            metadata (dict, optional): Metadata to store with the memory. Defaults to None.
            filters (dict, optional): Filters to apply to the search. Defaults to None.
            infer (bool, optional): Whether to infer the memories. Defaults to True.
            prompt (str, optional): Prompt to use for memory deduction. Defaults to None.

        Returns:
            dict: A dictionary containing the result of the memory addition operation.
            result: dict of affected events with each dict has the following key:
              'memories': affected memories
              'graph': affected graph memories

              'memories' and 'graph' is a dict, each with following subkeys:
                'add': added memory
                'update': updated memory
                'delete': deleted memory


        """
        if metadata is None:
            metadata = {}

        filters = filters or {}
        if user_id:
            filters["user_id"] = metadata["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = metadata["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = metadata["run_id"] = run_id

        if not any(key in filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError("One of the filters: user_id, agent_id or run_id is required!")

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        if self.config.llm.config.get("enable_vision"):
            messages = parse_vision_messages(messages, self.llm, self.config.llm.config.get("vision_details"))
        else:
            messages = parse_vision_messages(messages)

        # Extract role references from messages before processing
        role_references_result = self._extract_role_references(messages, filters)
        
        # If role references were found, attempt to resolve them
        if role_references_result.get("references"):
            resolution_result = self._resolve_role_references(
                messages, 
                role_references_result["references"], 
                filters
            )
            
            # If clarification is needed, return immediately without writing to memory
            if resolution_result["status"] == "clarification_needed":
                logger.info("Returning clarification request instead of storing memory")
                return resolution_result["clarification"]
            
            # If resolved, rewrite messages with canonical names
            if resolution_result["status"] == "resolved":
                messages = self._rewrite_messages_with_resolved_names(
                    messages, 
                    resolution_result["resolved_references"]
                )

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future1 = executor.submit(self._add_to_vector_store, messages, metadata, filters, infer)
            future2 = executor.submit(self._add_to_graph, messages, filters)

            concurrent.futures.wait([future1, future2])

            vector_store_result = future1.result()
            graph_result = future2.result()

        if self.api_version == "v1.0":
            warnings.warn(
                "The current add API output format is deprecated. "
                "To use the latest format, set `api_version='v1.1'`. "
                "The current format will be removed in mem0ai 1.1.0 and later versions.",
                category=DeprecationWarning,
                stacklevel=2,
            )
            return vector_store_result

        if self.enable_graph:
            return {
                "results": vector_store_result,
                "relations": graph_result,
            }

        return {"results": vector_store_result}

    def _add_to_vector_store(self, messages, metadata, filters, infer):
        if not infer:
            returned_memories = []
            for message in messages:
                if message["role"] != "system":
                    message_embeddings = self.embedding_model.embed(message["content"], "add")
                    memory_id = self._create_memory(message["content"], message_embeddings, metadata)
                    returned_memories.append({"id": memory_id, "memory": message["content"], "event": "ADD"})
            return returned_memories

        parsed_messages = parse_messages(messages)

        if self.custom_fact_extraction_prompt:
            system_prompt = self.custom_fact_extraction_prompt
            user_prompt = f"Input:\n{parsed_messages}"
        else:
            system_prompt, user_prompt = get_fact_retrieval_messages(parsed_messages)

        response = self.llm.generate_response(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )

        try:
            response = remove_code_blocks(response)
            new_retrieved_facts = json.loads(response)["facts"]
        except Exception as e:
            logging.error(f"Error in new_retrieved_facts: {e}")
            new_retrieved_facts = []

        retrieved_old_memory = []
        new_message_embeddings = {}
        for new_mem in new_retrieved_facts:
            messages_embeddings = self.embedding_model.embed(new_mem, "add")
            new_message_embeddings[new_mem] = messages_embeddings
            existing_memories = self.vector_store.search(
                query=new_mem,
                vectors=messages_embeddings,
                limit=5,
                filters=filters,
            )
            for mem in existing_memories:
                retrieved_old_memory.append({"id": mem.id, "text": mem.payload["data"]})
        unique_data = {}
        for item in retrieved_old_memory:
            unique_data[item["id"]] = item
        retrieved_old_memory = list(unique_data.values())
        logging.info(f"Total existing memories: {len(retrieved_old_memory)}")

        # mapping UUIDs with integers for handling UUID hallucinations
        temp_uuid_mapping = {}
        for idx, item in enumerate(retrieved_old_memory):
            temp_uuid_mapping[str(idx)] = item["id"]
            retrieved_old_memory[idx]["id"] = str(idx)

        function_calling_prompt = get_update_memory_messages(
            retrieved_old_memory, new_retrieved_facts, self.custom_update_memory_prompt
        )

        try:
            new_memories_with_actions = self.llm.generate_response(
                messages=[{"role": "user", "content": function_calling_prompt}],
                response_format={"type": "json_object"},
            )
        except Exception as e:
            logging.error(f"Error in new_memories_with_actions: {e}")
            new_memories_with_actions = []

        try:
            new_memories_with_actions = remove_code_blocks(new_memories_with_actions)
            new_memories_with_actions = json.loads(new_memories_with_actions)
        except Exception as e:
            logging.error(f"Invalid JSON response: {e}")
            new_memories_with_actions = []

        returned_memories = []
        try:
            for resp in new_memories_with_actions.get("memory", []):
                logging.info(resp)
                try:
                    if not resp.get("text"):
                        logging.info("Skipping memory entry because of empty `text` field.")
                        continue
                    elif resp.get("event") == "ADD":
                        memory_id = self._create_memory(
                            data=resp.get("text"), existing_embeddings=new_message_embeddings, metadata=metadata
                        )
                        returned_memories.append(
                            {
                                "id": memory_id,
                                "memory": resp.get("text"),
                                "event": resp.get("event"),
                            }
                        )
                    elif resp.get("event") == "UPDATE":
                        self._update_memory(
                            memory_id=temp_uuid_mapping[resp["id"]],
                            data=resp.get("text"),
                            existing_embeddings=new_message_embeddings,
                            metadata=metadata,
                        )
                        returned_memories.append(
                            {
                                "id": temp_uuid_mapping[resp.get("id")],
                                "memory": resp.get("text"),
                                "event": resp.get("event"),
                                "previous_memory": resp.get("old_memory"),
                            }
                        )
                    elif resp.get("event") == "DELETE":
                        self._delete_memory(memory_id=temp_uuid_mapping[resp.get("id")])
                        returned_memories.append(
                            {
                                "id": temp_uuid_mapping[resp.get("id")],
                                "memory": resp.get("text"),
                                "event": resp.get("event"),
                            }
                        )
                    elif resp.get("event") == "NONE":
                        logging.info("NOOP for Memory.")
                except Exception as e:
                    logging.error(f"Error in new_memories_with_actions: {e}")
        except Exception as e:
            logging.error(f"Error in new_memories_with_actions: {e}")

        capture_event("mem0.add", self, {"version": self.api_version, "keys": list(filters.keys())})

        return returned_memories

    def _add_to_graph(self, messages, filters):
        added_entities = []
        if self.enable_graph:
            if filters.get("user_id") is None:
                filters["user_id"] = "user"

            data = "\n".join([msg["content"] for msg in messages if "content" in msg and msg["role"] != "system"])
            added_entities = self.graph.add(data, filters)

        return added_entities

    def get(self, memory_id):
        """
        Retrieve a memory by ID.

        Args:
            memory_id (str): ID of the memory to retrieve.

        Returns:
            dict: Retrieved memory.
        """
        capture_event("mem0.get", self, {"memory_id": memory_id})
        memory = self.vector_store.get(vector_id=memory_id)
        if not memory:
            return None

        filters = {key: memory.payload[key] for key in ["user_id", "agent_id", "run_id"] if memory.payload.get(key)}

        # Prepare base memory item
        memory_item = MemoryItem(
            id=memory.id,
            memory=memory.payload["data"],
            hash=memory.payload.get("hash"),
            created_at=memory.payload.get("created_at"),
            updated_at=memory.payload.get("updated_at"),
        ).model_dump(exclude={"score"})

        # Add metadata if there are additional keys
        excluded_keys = {"user_id", "agent_id", "run_id", "hash", "data", "created_at", "updated_at", "id"}
        additional_metadata = {k: v for k, v in memory.payload.items() if k not in excluded_keys}
        if additional_metadata:
            memory_item["metadata"] = additional_metadata

        result = {**memory_item, **filters}

        return result

    def get_all(self, user_id=None, agent_id=None, run_id=None, limit=100):
        """
        List all memories.

        Returns:
            list: List of all memories.
        """
        filters = {}
        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id

        capture_event("mem0.get_all", self, {"limit": limit, "keys": list(filters.keys())})

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_memories = executor.submit(self._get_all_from_vector_store, filters, limit)
            future_graph_entities = executor.submit(self.graph.get_all, filters, limit) if self.enable_graph else None

            concurrent.futures.wait(
                [future_memories, future_graph_entities] if future_graph_entities else [future_memories]
            )

            all_memories = future_memories.result()
            graph_entities = future_graph_entities.result() if future_graph_entities else None

        if self.enable_graph:
            return {"results": all_memories, "relations": graph_entities}

        if self.api_version == "v1.0":
            warnings.warn(
                "The current get_all API output format is deprecated. "
                "To use the latest format, set `api_version='v1.1'`. "
                "The current format will be removed in mem0ai 1.1.0 and later versions.",
                category=DeprecationWarning,
                stacklevel=2,
            )
            return all_memories
        else:
            return {"results": all_memories}

    def _get_all_from_vector_store(self, filters, limit):
        memories = self.vector_store.list(filters=filters, limit=limit)

        excluded_keys = {
            "user_id",
            "agent_id",
            "run_id",
            "hash",
            "data",
            "created_at",
            "updated_at",
            "id",
        }
        all_memories = [
            {
                **MemoryItem(
                    id=mem.id,
                    memory=mem.payload["data"],
                    hash=mem.payload.get("hash"),
                    created_at=mem.payload.get("created_at"),
                    updated_at=mem.payload.get("updated_at"),
                ).model_dump(exclude={"score"}),
                **{key: mem.payload[key] for key in ["user_id", "agent_id", "run_id"] if key in mem.payload},
                **(
                    {"metadata": {k: v for k, v in mem.payload.items() if k not in excluded_keys}}
                    if any(k for k in mem.payload if k not in excluded_keys)
                    else {}
                ),
            }
            for mem in memories[0]
        ]
        return all_memories

    def search(self, query, user_id=None, agent_id=None, run_id=None, limit=100, filters=None):
        """
        Search for memories.

        Args:
            query (str): Query to search for.
            user_id (str, optional): ID of the user to search for. Defaults to None.
            agent_id (str, optional): ID of the agent to search for. Defaults to None.
            run_id (str, optional): ID of the run to search for. Defaults to None.
            limit (int, optional): Limit the number of results. Defaults to 100.
            filters (dict, optional): Filters to apply to the search. Defaults to None.

        Returns:
            list: List of search results.
        """
        filters = filters or {}
        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id

        if not any(key in filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError("One of the filters: user_id, agent_id or run_id is required!")

        capture_event(
            "mem0.search",
            self,
            {"limit": limit, "version": self.api_version, "keys": list(filters.keys())},
        )

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_memories = executor.submit(self._search_vector_store, query, filters, limit)
            future_graph_entities = (
                executor.submit(self.graph.search, query, filters, limit) if self.enable_graph else None
            )

            concurrent.futures.wait(
                [future_memories, future_graph_entities] if future_graph_entities else [future_memories]
            )

            original_memories = future_memories.result()
            graph_entities = future_graph_entities.result() if future_graph_entities else None

        if self.enable_graph:
            return {"results": original_memories, "relations": graph_entities}

        if self.api_version == "v1.0":
            warnings.warn(
                "The current get_all API output format is deprecated. "
                "To use the latest format, set `api_version='v1.1'`. "
                "The current format will be removed in mem0ai 1.1.0 and later versions.",
                category=DeprecationWarning,
                stacklevel=2,
            )
            return original_memories
        else:
            return {"results": original_memories}

    def _search_vector_store(self, query, filters, limit):
        embeddings = self.embedding_model.embed(query, "search")
        memories = self.vector_store.search(query=query, vectors=embeddings, limit=limit, filters=filters)

        excluded_keys = {
            "user_id",
            "agent_id",
            "run_id",
            "hash",
            "data",
            "created_at",
            "updated_at",
            "id",
        }

        original_memories = [
            {
                **MemoryItem(
                    id=mem.id,
                    memory=mem.payload["data"],
                    hash=mem.payload.get("hash"),
                    created_at=mem.payload.get("created_at"),
                    updated_at=mem.payload.get("updated_at"),
                    score=mem.score,
                ).model_dump(),
                **{key: mem.payload[key] for key in ["user_id", "agent_id", "run_id"] if key in mem.payload},
                **(
                    {"metadata": {k: v for k, v in mem.payload.items() if k not in excluded_keys}}
                    if any(k for k in mem.payload if k not in excluded_keys)
                    else {}
                ),
            }
            for mem in memories
        ]

        return original_memories

    def update(self, memory_id, data):
        """
        Update a memory by ID.

        Args:
            memory_id (str): ID of the memory to update.
            data (dict): Data to update the memory with.

        Returns:
            dict: Updated memory.
        """
        capture_event("mem0.update", self, {"memory_id": memory_id})

        existing_embeddings = {data: self.embedding_model.embed(data, "update")}

        self._update_memory(memory_id, data, existing_embeddings)
        return {"message": "Memory updated successfully!"}

    def delete(self, memory_id):
        """
        Delete a memory by ID.

        Args:
            memory_id (str): ID of the memory to delete.
        """
        capture_event("mem0.delete", self, {"memory_id": memory_id})
        self._delete_memory(memory_id)
        return {"message": "Memory deleted successfully!"}

    def delete_all(self, user_id=None, agent_id=None, run_id=None):
        """
        Delete all memories.

        Args:
            user_id (str, optional): ID of the user to delete memories for. Defaults to None.
            agent_id (str, optional): ID of the agent to delete memories for. Defaults to None.
            run_id (str, optional): ID of the run to delete memories for. Defaults to None.
        """
        filters = {}
        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id

        if not filters:
            raise ValueError(
                "At least one filter is required to delete all memories. If you want to delete all memories, use the `reset()` method."
            )

        capture_event("mem0.delete_all", self, {"keys": list(filters.keys())})
        memories = self.vector_store.list(filters=filters)[0]
        for memory in memories:
            self._delete_memory(memory.id)

        logger.info(f"Deleted {len(memories)} memories")

        if self.enable_graph:
            self.graph.delete_all(filters)

        return {"message": "Memories deleted successfully!"}

    def history(self, memory_id):
        """
        Get the history of changes for a memory by ID.

        Args:
            memory_id (str): ID of the memory to get history for.

        Returns:
            list: List of changes for the memory.
        """
        capture_event("mem0.history", self, {"memory_id": memory_id})
        return self.db.get_history(memory_id)

    def _create_memory(self, data, existing_embeddings, metadata=None):
        logging.info(f"Creating memory with {data=}")
        if data in existing_embeddings:
            embeddings = existing_embeddings[data]
        else:
            embeddings = self.embedding_model.embed(data, "add")
        memory_id = str(uuid.uuid4())
        metadata = metadata or {}
        metadata["data"] = data
        metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        metadata["created_at"] = datetime.now(pytz.utc).strftime('%Y-%m-%d %H:%M:%S')

        self.vector_store.insert(
            vectors=[embeddings],
            ids=[memory_id],
            payloads=[metadata],
        )
        self.db.add_history(memory_id, None, data, "ADD", created_at=metadata["created_at"])
        capture_event("mem0._create_memory", self, {"memory_id": memory_id})
        return memory_id

    def _update_memory(self, memory_id, data, existing_embeddings, metadata=None):
        logger.info(f"Updating memory with {data=}")

        try:
            existing_memory = self.vector_store.get(vector_id=memory_id)
        except Exception:
            raise ValueError(f"Error getting memory with ID {memory_id}. Please provide a valid 'memory_id'")
        prev_value = existing_memory.payload.get("data")

        new_metadata = metadata or {}
        new_metadata["data"] = data
        new_metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        new_metadata["created_at"] = existing_memory.payload.get("created_at")
        new_metadata["updated_at"] = datetime.now(pytz.utc).strftime('%Y-%m-%d %H:%M:%S')

        if "user_id" in existing_memory.payload:
            new_metadata["user_id"] = existing_memory.payload["user_id"]
        if "agent_id" in existing_memory.payload:
            new_metadata["agent_id"] = existing_memory.payload["agent_id"]
        if "run_id" in existing_memory.payload:
            new_metadata["run_id"] = existing_memory.payload["run_id"]

        if data in existing_embeddings:
            embeddings = existing_embeddings[data]
        else:
            embeddings = self.embedding_model.embed(data, "update")
        self.vector_store.update(
            vector_id=memory_id,
            vector=embeddings,
            payload=new_metadata,
        )
        logger.info(f"Updating memory with ID {memory_id=} with {data=}")
        self.db.add_history(
            memory_id,
            prev_value,
            data,
            "UPDATE",
            created_at=new_metadata["created_at"],
            updated_at=new_metadata["updated_at"],
        )
        capture_event("mem0._update_memory", self, {"memory_id": memory_id})
        return memory_id

    def _delete_memory(self, memory_id):
        logging.info(f"Deleting memory with {memory_id=}")
        existing_memory = self.vector_store.get(vector_id=memory_id)
        prev_value = existing_memory.payload["data"]
        self.vector_store.delete(vector_id=memory_id)
        self.db.add_history(memory_id, prev_value, None, "DELETE", is_deleted=1)
        capture_event("mem0._delete_memory", self, {"memory_id": memory_id})
        return memory_id

    def reset(self):
        """
        Reset the memory store.
        """
        logger.warning("Resetting all memories")
        self.vector_store.delete_col()
        self.vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, self.config.vector_store.config
        )
        self.db.reset()
        capture_event("mem0.reset", self)

    def chat(self, query):
        raise NotImplementedError("Chat function not implemented yet.")
