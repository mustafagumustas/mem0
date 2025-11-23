UPDATE_MEMORY_TOOL_GRAPH = {
    "type": "function",
    "function": {
        "name": "update_graph_memory",
        "description": "Update the relationship key of an existing graph memory based on new information. This function should be called when there's a need to modify an existing relationship in the knowledge graph. The update should only be performed if the new information is more recent, more accurate, or provides additional context compared to the existing information. The source and destination nodes of the relationship must remain the same as in the existing graph memory; only the relationship itself can be updated.",
        "parameters": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "The identifier of the source node in the relationship to be updated. This should match an existing node in the graph.",
                },
                "destination": {
                    "type": "string",
                    "description": "The identifier of the destination node in the relationship to be updated. This should match an existing node in the graph.",
                },
                "relationship": {
                    "type": "string",
                    "description": "The new or updated relationship between the source and destination nodes. This should be a concise, clear description of how the two nodes are connected.",
                },
                "weight": {
                    "type": "string",
                    "enum": ["ignored", "peripheral", "transitional", "relevant", "ritualistic", "important", "core_identity", "infatuation", "devotion", "obsession", "repressed", "negative_core"],
                    "description": "The significance category of this relationship to the user.",
                },
                "is_uncertain": {
                    "type": "boolean",
                    "description": "Indicates whether this relationship is uncertain or speculative.",
                },
                "status": {
                    "type": "string",
                    "description": "The current status of the relationship (e.g., 'active', 'inactive', 'pending').",
                },
                "start_date": {
                    "type": "string",
                    "description": "The date when this relationship started, in ISO format (YYYY-MM-DD).",
                },
                "end_date": {
                    "type": "string",
                    "description": "The date when this relationship ended (if applicable), in ISO format (YYYY-MM-DD).",
                },
                "emotion": {
                    "type": "string",
                    "description": "The emotional undertone or feeling the user expresses about this specific relationship. Capture the user's attitude, sentiment, or emotional quality when mentioning this relationship. Use descriptive emotion words that reflect intensity and nuance. Use 'neutral' only if no emotional context is detectable.",
                },
                "last_mentioned": {
                    "type": "string",
                    "description": "The timestamp when this relationship was last mentioned or referenced, in ISO format.",
                },
                "usage_count": {
                    "type": "integer",
                    "description": "The number of times this relationship has been referenced or mentioned (starts at 1).",
                },
            },
            "required": ["source", "destination", "relationship"],
            "additionalProperties": False,
        },
    },
}

ADD_MEMORY_TOOL_GRAPH = {
    "type": "function",
    "function": {
        "name": "add_graph_memory",
        "description": "Add a new graph memory to the knowledge graph. This function creates a new relationship between two nodes, potentially creating new nodes if they don't exist.",
        "parameters": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "The identifier of the source node in the new relationship. This can be an existing node or a new node to be created.",
                },
                "destination": {
                    "type": "string",
                    "description": "The identifier of the destination node in the new relationship. This can be an existing node or a new node to be created.",
                },
                "relationship": {
                    "type": "string",
                    "description": "The type of relationship between the source and destination nodes. This should be a concise, clear description of how the two nodes are connected.",
                },
                "source_type": {
                    "type": "string",
                    "description": "The type or category of the source node. This helps in classifying and organizing nodes in the graph.",
                },
                "destination_type": {
                    "type": "string",
                    "description": "The type or category of the destination node. This helps in classifying and organizing nodes in the graph.",
                },
                "weight": {
                    "type": "string",
                    "enum": ["ignored", "peripheral", "transitional", "relevant", "ritualistic", "important", "core_identity", "infatuation", "devotion", "obsession", "repressed", "negative_core"],
                    "description": "The significance category of this relationship to the user.",
                },
                "is_uncertain": {
                    "type": "boolean",
                    "description": "Indicates whether this relationship is uncertain or speculative.",
                },
                "status": {
                    "type": "string",
                    "description": "The current status of the relationship (e.g., 'active', 'inactive', 'pending').",
                },
                "start_date": {
                    "type": "string",
                    "description": "The date when this relationship started, in ISO format (YYYY-MM-DD).",
                },
                "end_date": {
                    "type": "string",
                    "description": "The date when this relationship ended (if applicable), in ISO format (YYYY-MM-DD).",
                },
                "emotion": {
                    "type": "string",
                    "description": "The emotional undertone or feeling the user expresses about this specific relationship. Capture the user's attitude, sentiment, or emotional quality when mentioning this relationship. Use descriptive emotion words that reflect intensity and nuance. Use 'neutral' only if no emotional context is detectable.",
                },
                "last_mentioned": {
                    "type": "string",
                    "description": "The timestamp when this relationship was last mentioned or referenced, in ISO format.",
                },
                "usage_count": {
                    "type": "integer",
                    "description": "The number of times this relationship has been referenced or mentioned (starts at 1).",
                },
            },
            "required": [
                "source",
                "destination",
                "relationship",
                "source_type",
                "destination_type",
            ],
            "additionalProperties": False,
        },
    },
}


NOOP_TOOL = {
    "type": "function",
    "function": {
        "name": "noop",
        "description": "No operation should be performed to the graph entities. This function is called when the system determines that no changes or additions are necessary based on the current input or context. It serves as a placeholder action when no other actions are required, ensuring that the system can explicitly acknowledge situations where no modifications to the graph are needed.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
}


RELATIONS_TOOL = {
    "type": "function",
    "function": {
        "name": "establish_relationships",
        "description": "Establish relationships among the entities based on the provided text.",
        "parameters": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": {
                                "type": "string",
                                "description": "The source entity of the relationship.",
                            },
                            "relationship": {
                                "type": "string",
                                "description": "The relationship between the source and destination entities.",
                            },
                            "destination": {
                                "type": "string",
                                "description": "The destination entity of the relationship.",
                            },
                            "owner_person_name": {
                                "type": ["string", "null"],
                                "description": "normalized person name that owns this fact.",
                            },
                            "weight": {
                                "type": "string",
                                "enum": ["ignored", "peripheral", "transitional", "relevant", "ritualistic", "important", "core_identity", "infatuation", "devotion", "obsession", "repressed", "negative_core"],
                                "description": "The significance category of this relationship to the user.",
                            },
                            "is_uncertain": {
                                "type": "boolean",
                                "description": "Indicates whether this relationship is uncertain or speculative.",
                            },
                            "status": {
                                "type": "string",
                                "description": "The current status of the relationship (e.g., 'active', 'inactive', 'pending').",
                            },
                            "start_date": {
                                "type": "string",
                                "description": "The date when this relationship started, in ISO format (YYYY-MM-DD).",
                            },
                            "end_date": {
                                "type": "string",
                                "description": "The date when this relationship ended (if applicable), in ISO format (YYYY-MM-DD).",
                            },
                            "emotion": {
                                "type": "string",
                                "description": "The emotional undertone or feeling the user expresses about this specific relationship. Capture the user's attitude, sentiment, or emotional quality when mentioning this relationship. Use descriptive emotion words that reflect intensity and nuance. Use 'neutral' only if no emotional context is detectable.",
                            },
                            "last_mentioned": {
                                "type": "string",
                                "description": "The timestamp when this relationship was last mentioned or referenced, in ISO format.",
                            },
                            "usage_count": {
                                "type": "integer",
                                "description": "The number of times this relationship has been referenced or mentioned (starts at 1).",
                            },
                        },
                        "required": [
                            "source",
                            "relationship",
                            "destination",
                            "owner_person_name",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["entities"],
            "additionalProperties": False,
        },
    },
}


EXTRACT_ENTITIES_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_entities",
        "description": "Extract entities and their types from the text.",
        "parameters": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "entity": {
                                "type": "string",
                                "description": "The name or identifier of the entity.",
                            },
                            "entity_type": {
                                "type": "string",
                                "description": "The type or category of the entity.",
                            },
                        },
                        "required": ["entity", "entity_type"],
                        "additionalProperties": False,
                    },
                    "description": "An array of entities with their types.",
                }
            },
            "required": ["entities"],
            "additionalProperties": False,
        },
    },
}

UPDATE_MEMORY_STRUCT_TOOL_GRAPH = {
    "type": "function",
    "function": {
        "name": "update_graph_memory",
        "description": "Update the relationship key of an existing graph memory based on new information. This function should be called when there's a need to modify an existing relationship in the knowledge graph. The update should only be performed if the new information is more recent, more accurate, or provides additional context compared to the existing information. The source and destination nodes of the relationship must remain the same as in the existing graph memory; only the relationship itself can be updated.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "The identifier of the source node in the relationship to be updated. This should match an existing node in the graph.",
                },
                "destination": {
                    "type": "string",
                    "description": "The identifier of the destination node in the relationship to be updated. This should match an existing node in the graph.",
                },
                "relationship": {
                    "type": "string",
                    "description": "The new or updated relationship between the source and destination nodes. This should be a concise, clear description of how the two nodes are connected.",
                },
                "weight": {
                    "type": "string",
                    "enum": ["ignored", "peripheral", "transitional", "relevant", "ritualistic", "important", "core_identity", "infatuation", "devotion", "obsession", "repressed", "negative_core"],
                    "description": "The significance category of this relationship to the user.",
                },
                "is_uncertain": {
                    "type": "boolean",
                    "description": "Indicates whether this relationship is uncertain or speculative.",
                },
                "status": {
                    "type": "string",
                    "description": "The current status of the relationship (e.g., 'active', 'inactive', 'pending').",
                },
                "start_date": {
                    "type": "string",
                    "description": "The date when this relationship started, in ISO format (YYYY-MM-DD).",
                },
                "end_date": {
                    "type": "string",
                    "description": "The date when this relationship ended (if applicable), in ISO format (YYYY-MM-DD).",
                },
                "emotion": {
                    "type": "string",
                    "description": "The emotional undertone or feeling the user expresses about this specific relationship. Capture the user's attitude, sentiment, or emotional quality when mentioning this relationship. Use descriptive emotion words that reflect intensity and nuance. Use 'neutral' only if no emotional context is detectable.",
                },
                "last_mentioned": {
                    "type": "string",
                    "description": "The timestamp when this relationship was last mentioned or referenced, in ISO format.",
                },
                "usage_count": {
                    "type": "integer",
                    "description": "The number of times this relationship has been referenced or mentioned (starts at 1).",
                },
            },
            "required": ["source", "destination", "relationship"],
            "additionalProperties": False,
        },
    },
}

ADD_MEMORY_STRUCT_TOOL_GRAPH = {
    "type": "function",
    "function": {
        "name": "add_graph_memory",
        "description": "Add a new graph memory to the knowledge graph. This function creates a new relationship between two nodes, potentially creating new nodes if they don't exist.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "The identifier of the source node in the new relationship. This can be an existing node or a new node to be created.",
                },
                "destination": {
                    "type": "string",
                    "description": "The identifier of the destination node in the new relationship. This can be an existing node or a new node to be created.",
                },
                "relationship": {
                    "type": "string",
                    "description": "The type of relationship between the source and destination nodes. This should be a concise, clear description of how the two nodes are connected.",
                },
                "source_type": {
                    "type": "string",
                    "description": "The type or category of the source node. This helps in classifying and organizing nodes in the graph.",
                },
                "destination_type": {
                    "type": "string",
                    "description": "The type or category of the destination node. This helps in classifying and organizing nodes in the graph.",
                },
                "weight": {
                    "type": "string",
                    "enum": ["ignored", "peripheral", "transitional", "relevant", "ritualistic", "important", "core_identity", "infatuation", "devotion", "obsession", "repressed", "negative_core"],
                    "description": "The significance category of this relationship to the user.",
                },
                "is_uncertain": {
                    "type": "boolean",
                    "description": "Indicates whether this relationship is uncertain or speculative.",
                },
                "status": {
                    "type": "string",
                    "description": "The current status of the relationship (e.g., 'active', 'inactive', 'pending').",
                },
                "start_date": {
                    "type": "string",
                    "description": "The date when this relationship started, in ISO format (YYYY-MM-DD).",
                },
                "end_date": {
                    "type": "string",
                    "description": "The date when this relationship ended (if applicable), in ISO format (YYYY-MM-DD).",
                },
                "emotion": {
                    "type": "string",
                    "description": "The emotional undertone or feeling the user expresses about this specific relationship. Capture the user's attitude, sentiment, or emotional quality when mentioning this relationship. Use descriptive emotion words that reflect intensity and nuance. Use 'neutral' only if no emotional context is detectable.",
                },
                "last_mentioned": {
                    "type": "string",
                    "description": "The timestamp when this relationship was last mentioned or referenced, in ISO format.",
                },
                "usage_count": {
                    "type": "integer",
                    "description": "The number of times this relationship has been referenced or mentioned (starts at 1).",
                },
            },
            "required": [
                "source",
                "destination",
                "relationship",
                "source_type",
                "destination_type",
            ],
            "additionalProperties": False,
        },
    },
}


NOOP_STRUCT_TOOL = {
    "type": "function",
    "function": {
        "name": "noop",
        "description": "No operation should be performed to the graph entities. This function is called when the system determines that no changes or additions are necessary based on the current input or context. It serves as a placeholder action when no other actions are required, ensuring that the system can explicitly acknowledge situations where no modifications to the graph are needed.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
}

RELATIONS_STRUCT_TOOL = {
    "type": "function",
    "function": {
        "name": "establish_relations",
        "description": "Establish relationships among the entities based on the provided text.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source_entity": {
                                "type": "string",
                                "description": "The source entity of the relationship.",
                            },
                            "relatationship": {
                                "type": "string",
                                "description": "The relationship between the source and destination entities.",
                            },
                            "destination_entity": {
                                "type": "string",
                                "description": "The destination entity of the relationship.",
                            },
                            "owner_person_name": {
                                "type": ["string", "null"],
                                "description": "normalized person name that owns this fact.",
                            },
                            "weight": {
                                "type": "string",
                                "enum": ["ignored", "peripheral", "transitional", "relevant", "ritualistic", "important", "core_identity", "infatuation", "devotion", "obsession", "repressed", "negative_core"],
                                "description": "The significance category of this relationship to the user.",
                            },
                            "is_uncertain": {
                                "type": "boolean",
                                "description": "Indicates whether this relationship is uncertain or speculative.",
                            },
                            "status": {
                                "type": "string",
                                "description": "The current status of the relationship (e.g., 'active', 'inactive', 'pending').",
                            },
                            "start_date": {
                                "type": "string",
                                "description": "The date when this relationship started, in ISO format (YYYY-MM-DD).",
                            },
                            "end_date": {
                                "type": "string",
                                "description": "The date when this relationship ended (if applicable), in ISO format (YYYY-MM-DD).",
                            },
                            "emotion": {
                                "type": "string",
                                "description": "The emotional undertone or feeling the user expresses about this specific relationship. Capture the user's attitude, sentiment, or emotional quality when mentioning this relationship. Use descriptive emotion words that reflect intensity and nuance. Use 'neutral' only if no emotional context is detectable.",
                            },
                            "last_mentioned": {
                                "type": "string",
                                "description": "The timestamp when this relationship was last mentioned or referenced, in ISO format.",
                            },
                            "usage_count": {
                                "type": "integer",
                                "description": "The number of times this relationship has been referenced or mentioned (starts at 1).",
                            },
                        },
                        "required": [
                            "source_entity",
                            "relatationship",
                            "destination_entity",
                            "owner_person_name",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["entities"],
            "additionalProperties": False,
        },
    },
}


EXTRACT_ENTITIES_STRUCT_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_entities",
        "description": "Extract entities and their types from the text.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "entity": {
                                "type": "string",
                                "description": "The name or identifier of the entity.",
                            },
                            "entity_type": {
                                "type": "string",
                                "description": "The type or category of the entity.",
                            },
                        },
                        "required": ["entity", "entity_type"],
                        "additionalProperties": False,
                    },
                    "description": "An array of entities with their types.",
                }
            },
            "required": ["entities"],
            "additionalProperties": False,
        },
    },
}

DELETE_MEMORY_STRUCT_TOOL_GRAPH = {
    "type": "function",
    "function": {
        "name": "delete_graph_memory",
        "description": "Delete the relationship between two nodes. This function deletes the existing relationship.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "The identifier of the source node in the relationship.",
                },
                "relationship": {
                    "type": "string",
                    "description": "The existing relationship between the source and destination nodes that needs to be deleted.",
                },
                "destination": {
                    "type": "string",
                    "description": "The identifier of the destination node in the relationship.",
                },
            },
            "required": [
                "source",
                "relationship",
                "destination",
            ],
            "additionalProperties": False,
        },
    },
}

DELETE_MEMORY_TOOL_GRAPH = {
    "type": "function",
    "function": {
        "name": "delete_graph_memory",
        "description": "Delete the relationship between two nodes. This function deletes the existing relationship.",
        "parameters": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "The identifier of the source node in the relationship.",
                },
                "relationship": {
                    "type": "string",
                    "description": "The existing relationship between the source and destination nodes that needs to be deleted.",
                },
                "destination": {
                    "type": "string",
                    "description": "The identifier of the destination node in the relationship.",
                },
            },
            "required": [
                "source",
                "relationship",
                "destination",
            ],
            "additionalProperties": False,
        },
    },
}

ANALYZE_RELATION_EVOLUTION_TOOL = {
    "type": "function",
    "function": {
        "name": "analyze_relation_evolution",
        "description": "Infer behavioral and emotional changes in a user's relationship with an entity based on recent conversation history and graph context.",
        "parameters": {
            "type": "object",
            "properties": {
                "weight": {
                    "type": "string",
                    "enum": ["ignored", "peripheral", "transitional", "relevant", "ritualistic", "important", "core_identity", "infatuation", "devotion", "obsession", "repressed", "negative_core"],
                    "description": "Updated significance category representing how much this relationship matters to the user."
                },
                "emotion": {
                    "type": "string",
                    "description": "Updated emotional tone of the relationship. For example, if a user expresses frustration about something they previously liked, the emotion could change from 'joy' to 'annoyance'."
                },
                "status": {
                    "type": "string",
                    "description": "Updated status: 'active', 'ended', or 'invalid'. For example, if a user says they've stopped a hobby, the status could become 'ended'."
                },
                "is_uncertain": {
                    "type": "boolean",
                    "description": "True if the emotional shift is unclear or speculative. This is for when you are not confident in the analysis."
                },
                "has_emotional_shift": {
                    "type": "boolean",
                    "description": "True if the user's emotional tone towards the entity has changed significantly."
                },
                "has_habit_changed": {
                    "type": "boolean",
                    "description": "True if the user's behavior or habits related to the entity have changed, e.g., frequency of mention."
                },
                "has_new_obsession": {
                    "type": "boolean",
                    "description": "True if a new interest seems to be displacing this entity in the user's focus."
                }
            },
            "required": ["weight", "emotion", "status", "is_uncertain", "has_emotional_shift", "has_habit_changed", "has_new_obsession"]
        }
    }
}
