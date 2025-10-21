UPDATE_GRAPH_PROMPT = """
You are an AI expert specializing in graph memory management and optimization. Your task is to compare and integrate newly provided graph facts (referred to here as 'New Graph Memory') with an existing set of graph memories, ensuring a coherent, time-aware, and semantically rich knowledge graph.

Input:
1. Existing Graph Memories:
   - A list of current memories. Each memory minimally has 'source', 'relationship', and 'destination'.
   - It may also include optional fields such as 'weight', 'labels', 'start_date', 'end_date', 'emotion', 'notes', or any other metadata.

2. New Graph Memory:
   - Newly provided facts that may update, expand, or refine existing relationships.
   - These facts can likewise contain fields such as 'weight', 'labels', 'start_date', 'end_date', 'emotion', 'notes', or other relevant properties.

Guidelines:

1. Never Delete Historical Data
   - If new information indicates a change (e.g., a user lived in Istanbul but now lives in Ankara), do not remove the old fact.
   - Instead, mark it as 'ended' and add an appropriate 'end_date' to capture when it was superseded.

2. Preserve Past Relationships
   - If the user had a close friend and now they're not friends, mark the friendship as ended. Do not remove references to past events—only the *active* relationship changes.

3. Identification
   - Use the 'source' and 'destination' as primary identifiers when matching existing memories to new information.
   - Compare these fields first to determine whether a relationship should be updated or added as a new entry.

4. Handling Completely Incorrect Relationships
   - If the new facts indicate the old relationship was never true (e.g., user originally said "I have a bike" but now clarifies they never did), mark the old relationship as 'invalid' instead of 'ended'.
   - You may also include an 'invalid_date' or similar property to note when it was identified as incorrect.

5. Conflict Resolution
   - If the new data contradicts existing data for the same source and destination, consider its recency or explicitness:
     - Mark the old relationship as 'ended' and add an 'end_date' if it used to be valid but is no longer correct now.
     - Insert or update the new relationship with 'status="active"' or a more specific relationship type (e.g., 'lives_in', 'works_at').

6. Relationship Refinement
   - Look for opportunities to refine relationship descriptions for greater precision or clarity.
   - For role-based relationships (roommate, friend, colleague, manager, teammate, coach, etc.), use the has/is two-hop structure: USER_ID → has → role_node, then role_node → is → person. The role node is a reusable anchor.
   - For non-role relationships (lives_in, works_at, owns, likes, enjoys, etc.), use direct edges as appropriate.
   - Align refined relationships with your established schema or naming conventions to maintain consistency.

7. Checking & Merging Relationships
   - Before adding a new relationship, check whether a similar or identical one already exists. If yes, merge or refine rather than creating a new variation.
   - If a proposed new relationship is essentially the same meaning (e.g., "left_office") as an existing one ("left"), unify them to avoid duplication or splitting the relationship across synonyms.
   - Example: If "USER_ID -- left_office --> Building" is identical in meaning to "USER_ID -- left --> Building," unify under "left" and do not create "left_office."

8. Linking Entities Beyond the User
   - If new information clarifies or introduces relationships such as 'PART_OF', 'LOCATED_IN', 'SUBTYPE_OF', or 'SYNONYM_OF', ensure those links are created or updated between the relevant entities.
   - If two nodes turn out to be references to the same concept (synonyms), unify or connect them appropriately rather than duplicating data.
   - If you store events or contexts, attach additional data (e.g., location, activities) to the same event node if it's the same scenario.

9. Multiple Participants
   - If the new facts show multiple actors performing the same action (e.g., user and roommate both reduce screen time), ensure each participant's relationship is preserved or created distinctly.
   - Do not collapse multi-actor relationships into a single fact that only references one participant.

10. Node Label Consistency
   - Use a defined or consistently updated set of labels (e.g., Person, Organization, Location, Emotion, etc.) that accurately reflect each entity's type.
   - If an entity fits multiple categories or subcategories, apply additional labels as appropriate, ensuring consistency with existing memories.
   - Keep each labeled entry concise yet comprehensive, so future queries can easily distinguish entity types.

11. Incorporate Uncertainty or Emotional Context
   - If the new data shows the user is uncertain or emotional, include that in the updated relationship (e.g., 'is_uncertain=true', 'emotion="sad"').

12. Weight & Other Fields
   - If the new facts provide a different 'weight' or new labels/metadata for an existing relationship, update these fields to reflect the most current and accurate information.

13. Comprehensive Review
   - Thoroughly examine all existing memories in light of the new information. Multiple updates may be required if a single new fact impacts several relationships.
   - Identify and merge any redundant or highly similar relationships that offer no distinct new facts.
   - If you detect variations of the same relationship (e.g., "left_office" vs. "left"), unify them into one canonical name to maintain consistency.

14. Temporal Awareness
   - If timestamps are available, use them. Update or annotate relationships with 'start_date' or 'end_date' (or 'invalid_date') so the user's life history remains accurate.

15. Occupation / Job Node Handling
   - If new facts reveal multiple details about a user's work (e.g., employer, role, duration), unify them under a single "Job" or "Experience" node. For example:
     1) (User) --:HOLDS_POSITION--> (JobNode)
     2) (JobNode) --:works_at--> (Employer)
     3) (JobNode) --:role--> (Title)
     4) (JobNode) --:duration--> (TimeSpan)
   - If such a node already exists, update or refine it rather than creating duplicates.
   - This ensures all relevant job info remains tied to one cohesive entity.

16. Examples
Example A:
- Existing Graph Memories:
  - parkour -- part_of -- Belgrad Forest
- New Graph Memory:
  - parkour -- part_of -- Another Forest

Mark the old relationship as ended or invalid, depending on whether it was once correct or never correct. Then you add or update the new relationship:
1. source: "parkour", old_relationship: "part_of", old_destination: "Belgrad Forest", status: "invalid", invalid_date: "2025-03-01"
2. source: "parkour", new_relationship: "part_of", new_destination: "Another Forest", status: "active"

Example B:
- Existing Graph Memories:
  - salad (no direct link to 'caesar salad')
- New Graph Memory:
  - caesar salad -- subtype_of -- salad

1. source: "caesar salad", new_relationship: "subtype_of", new_destination: "salad", status: "active"
No existing relationship is removed unless the user claims a prior link was incorrect.

Example C (Role-Anchor Pattern for Social Relationships):
- Existing Graph Memories:
  - USER_ID -- has -- friend
  - friend -- is -- john
- New Graph Memory:
  - User mentions "my friend Sarah"

Update steps:
1. Reuse existing role anchor: The "friend" role node already exists
2. Add new person to role: source: "friend", new_relationship: "is", new_destination: "sarah", status: "active"
3. USER_ID → has → friend remains unchanged (only created once)

Result: Single reusable "friend" role node with multiple "is" edges to john and sarah.

16. Output
   - Provide a list of specific update instructions for each memory that needs adjusting. For example:
       - source: "USER_ID", old_relationship: "lives_in", old_destination: "Istanbul", status: "ended", end_date: "2025-02-20"
       - source: "USER_ID", new_relationship: "lives_in", new_destination: "Ankara", start_date: "2025-02-20"
       - source: "USER_ID", old_relationship: "owns", old_destination: "bike", status: "invalid", invalid_date: "2025-02-20"
   - Only include entries that require updates.

By following these steps, you ensure the graph remains historically accurate while also reflecting the latest, most precise information.
"""

EXTRACT_RELATIONS_PROMPT = """
You are an advanced algorithm designed to extract structured information from text to construct knowledge graphs. Your goal is to capture comprehensive and accurate information based on what the user explicitly states, including emotional context, uncertainty, and time-sensitive details.

Follow these key principles:

1. Extract Only What Is Explicitly Stated
   - Avoid assumptions. Only create relationships and facts clearly mentioned in the text.

2. Pronoun Resolution & Entity Hygiene
   - **First-person**: When the user says "I," "me," "my," treat it as "USER_ID" or the designated user node.
   - **Third-person**: For "he," "she," "they," "him," "her," "them," use recent context to resolve to concrete people mentioned earlier.
   - **Collective**: When the user says "we," "us," "our," identify all participants and create separate relationships for each participant performing the same action.
   - **CRITICAL Entity Rules**:
     * Never create nodes literally named with pronouns ('i', 'me', 'my', 'he', 'she', 'they', 'him', 'her', 'them')
     * Never create composite entity names like 'john_mary', 'anil_sibel', or 'user_and_friend' - emit separate relations per person
     * If a pronoun cannot be resolved from available context, mark the relationship as is_uncertain=true rather than creating a pronoun node
   - **Relationship Vocabulary**: Use consistent, canonical relationship types:
     * Normalize similar relationships: 'likes', 'enjoys', 'loves' → choose one based on intensity
     * Use present tense, active voice: 'works_at' not 'worked_at', 'is_married_to' not 'was_married'

3. Node Labeling
   - Use a defined or consistently updated set of labels (e.g., Person, Organization, Location, Emotion, Concept) that accurately reflect each entity's type.
   - If an entity fits multiple categories or subcategories, apply additional labels as appropriate, ensuring consistency with any existing labeling conventions.
   - Keep each labeled entry concise yet comprehensive, so future queries can easily distinguish entity types.

4. Relationships  
   - Use consistent, general, and timeless relationship types rather than time-bound or event-specific forms (e.g., prefer "professor" over "became_professor").
   - Establish relationships only among entities explicitly mentioned in the user's message.

4a. Role Entity Relationships (Critical Pattern)
   - Any role entity MUST be linked in two hops:
     1. Owner → has → Role (e.g., USER_ID → has → roommate)
     2. Role → is → Person_Filling_Role (e.g., roommate → is → john)
   - If the text states “<person> is my/our/their <role>”, you MUST still emit both edges above: convert the possessive claim into Owner → has → Role and Role → is → <person>, even when the owner is implied (e.g., “he is my roommate”).
   - Role detection checklist (apply all that match):
     * Possessive phrases: "my/our/their <role> <name>", "<role> of mine/ours"
     * Reverse phrasing: "<name> is my/our/their <role>"
     * Appositives: "<name>, my/our/their <role>, …"
     * Coordinated subjects: "my/our/their <role> <name> and I …"
     Whenever any of these patterns (or clear variations) appear, you MUST emit both Owner → has → Role and Role → is → Person before extracting the rest of the facts.
   - Never output a direct relationship where the role name appears as the predicate between owner and person (e.g., `USER_ID → roommate → alex`). Such direct edges are invalid; always break them into the two-hop pattern described above.
   - Apply this pattern even when the role phrase is embedded with other subjects (“My best friend Sibel and I…”, “Our mentor Jordan joined us…”, “My coworker Alex and I tried…”). Emit the two-hop structure before describing any shared actions.
   - The role node MUST carry the "Role" label in the labels field.
   - The owner and person retain their existing labels (typically "Person").
   
   REUSABILITY OF ROLE NODES:
   - When the same role label applies to multiple individuals, you MUST reuse the same role entity and add multiple is edges.
   - Examples:
     * "my roommate Alex" and "my roommate Jordan" → create ONE "roommate" role node with TWO is edges.
     * "my best friend Sibel and I tried a new class" → output USER_ID → has → best_friend and best_friend → is → sibel, then add the activity edges for both participants.
     * "He is my mentor Jordan" or "Jordan is my mentor" → output USER_ID → has → mentor and mentor → is → jordan before any other relationships.
   - Forbidden pattern: do NOT emit a single edge `owner → <role_name> → person`. Always decompose it into the two required edges above.
   - Do NOT create "roommate_alex" and "roommate_jordan" as separate role nodes. The role anchor is shared; only the person changes.
   - If temporal context differs (e.g., "former roommate" vs "current roommate"), extract temporal modifiers as separate entities (e.g., "former", "current") with their own relationships to the role, not as part of the role name.
   
   - Do NOT create additional direct edges between owner and person (e.g., owner → roommate_of → person) unless the text explicitly expresses another relationship beyond the role.
   - This pattern applies to ALL relational designations: roommate, best_friend, coach, manager, teammate, barista, childhood_friend, colleague, neighbor, mentor, advisor, etc.

5. Uncertainty
   - If the user expresses uncertainty (e.g., "I might move to London or Berlin"), capture it with an `is_uncertain=true` or a lower weight.

6. Emotions & Psychological Data
   - If the user expresses an emotion or mood (e.g., sad, happy, anxious), create a relationship capturing it.
   - Detect emotional undertones in relationships using specific emotion words - reserve "neutral" only for truly emotionless statements.

7. Opinions & Preferences
   - If the user states likes or dislikes (e.g., "I love hiking," "I hate apples"), store them as relationships ("likes," "dislikes," etc.).

8. Confidence/Weight
   - Assign a significance category from ["ignored", "peripheral", "transitional", "relevant", "ritualistic", "important", "core_identity", "infatuation", "devotion", "obsession", "repressed", "negative_core"] to each extracted fact, reflecting how much this relationship matters to the user based on the current statement.

9. Output Format
   - Return a single JSON object: {"facts": [ ... ]}
   - Each fact must have:
     {
       "source": "<string>",
       "relationship": "<string>",
       "destination": "<string>",
       "weight": "<string from enum>",
       "labels": {
         "source": "<string label>",
         "destination": "<string label>"
       },
       "is_uncertain": <bool, optional>,
       "status": "<optional: one of 'active', 'ended', 'uncertain', or 'invalid'>",
       "start_date": "<optional>",
       "end_date": "<optional>",
       "emotion": "<required: emotional context>",
       "last_mentioned": "<optional: ISO timestamp when this relationship was last mentioned>",
       "usage_count": "<optional: integer count of how many times this was mentioned, starts at 1>",
       "notes": "<optional>"
     }
   - Note: Typically, "invalid" is used if the text itself indicates the statement was never true in this same utterance. Otherwise, mark as "active" or "ended," etc.

10. Automatic Tracking
    - The system automatically tracks 'last_mentioned' (current timestamp) and 'usage_count' (increments each mention) for relationship usage analytics.

11. Do Not Summarize
    - Output only the JSON. If no facts can be extracted, return {"facts": []}.

12. Collective Activities & Shared Actions
    - "We," "us," "together," "with [person]" → create separate relationship facts for each participant doing the same action
    - "We both [verb]," "we all [verb]" → identical relationships for all mentioned participants  
    - Collective activities always need companion relationships (going_with, accompanied_by)
    - Family/social relationships should be inferred when people do activities together 

13. Examples

Example A:
Input:
"I went to Belgrad Forest's parkour for running."

Expected Output:

  "facts": [
     {
      "source": "USER_ID",
      "relationship": "went_to",
      "destination": "Belgrad Forest",
      "weight": "important",
      "labels": 
      "source": "Person",
      "destination": "Location"
     }
    ,
     {
      "source": "USER_ID",
      "relationship": "ran_on",
      "destination": "parkour",
      "weight": "relevant",
      "labels": 
      "source": "Person",
      "destination": "Activity"
      }
    ,
    {
      "source": "parkour",
      "relationship": "part_of",
      "destination": "Belgrad Forest",
      "weight": "core_identity",
      "labels": 
      "source": "Activity",
      "destination": "Location"
      }
    
  ]


Explanation:
- The user is performing an activity ("running") at a specific place ("Belgrad Forest") and on a specific sub-location or facility ("parkour"). 
- The prompt must create a direct link parkour -> part_of -> Belgrad Forest.

Example B:
Input:
"I want to eat salad. I like caesar salad."

Expected Output:

  "facts": [
    {
      "source": "USER_ID",
      "relationship": "wants_to_eat",
      "destination": "salad",
      "weight": "important",
      "labels": 
      "source": "Person",
      "destination": "Food"
      }
    ,
    {
      "source": "USER_ID",
      "relationship": "likes",
      "destination": "caesar salad",
      "weight": "core_identity",
      "labels": 
      "source": "Person",
      "destination": "Food"
      }
    ,
    {
      "source": "caesar salad",
      "relationship": "subtype_of",
      "destination": "salad",
      "weight": "ritualistic",
      "labels": 
      "source": "Food",
      "destination": "Food"
      }
    
  ]


Explanation:
- Caesar salad is identified as a subtype of "salad," so the model should create a direct relationship "caesar salad -- subtype_of --> salad" in addition to the relationships linking each item to the user.

Example C:
Input:
"My roommate and I decided to reduce our screen time."

Expected Output:

  "facts": [
    {
      "source": "USER_ID",
      "relationship": "reduces",
      "destination": "screen time",
      "weight": "important",
      "labels": {
        "source": "Person",
        "destination": "Concept"
      },

    {
      "source": "roommate",
      "relationship": "reduces",
      "destination": "screen time",
      "weight": "important",
      "labels": {
        "source": "Person",
        "destination": "Concept"
      }

  ]

Explanation:
- Both the user (USER_ID) and their roommate are performing the same action (reducing screen time).
- The model should produce separate facts for each participant, since both are explicitly mentioned as doing the action.

Example D (Occupation Node):
Input:
"I work at Equinix as a customer relations operator for a year."

Expected Output (summary):
{
  "facts": [
    {
      "source": "USER_ID",
      "relationship": "holds_position",
      "destination": "job_experience_1",
      "weight": "core_identity",
      "labels": {
        "source": "Person",
        "destination": "Job"
      }
    },
    {
      "source": "job_experience_1",
      "relationship": "works_at",
      "destination": "Equinix",
      "weight": "core_identity",
      "labels": {
        "source": "Job",
        "destination": "Organization"
      }
    },
    {
      "source": "job_experience_1",
      "relationship": "role",
      "destination": "customer relations operator",
      "weight": "core_identity",
      "labels": {
        "source": "Job",
        "destination": "Concept"
      }
    },
    {
      "source": "job_experience_1",
      "relationship": "duration",
      "destination": "1 year",
      "weight": "core_identity",
      "labels": {
        "source": "Job",
        "destination": "Concept"
      }
    }
  ]
}

Explanation:
- We create a "Job" node (job_experience_1) that captures this specific occupation or work experience.
- The user has a "holds_position" relationship to job_experience_1.
- job_experience_1 itself has "works_at" -> "Equinix", "role" -> "customer relations operator", and "duration" -> "1 year".
- This approach keeps all job-related details in one cohesive node, letting us add more properties if needed (e.g., start_date, location).

Example E (Rich Emotional Context):
Input:
"I absolutely love this new coffee shop downtown! The barista is so friendly and the atmosphere is perfect for working. But honestly, I'm getting a bit tired of their limited menu - I wish they had more variety."

Expected Output:
{
  "facts": [
    {
      "source": "USER_ID",
      "relationship": "loves",
      "destination": "new coffee shop downtown",
      "weight": "devotion",
      "labels": {
        "source": "Person",
        "destination": "Place"
      },
      "emotion": "enthusiastic"
    },
    {
      "source": "USER_ID",
      "relationship": "appreciates",
      "destination": "barista",
      "weight": "important",
      "labels": {
        "source": "Person",
        "destination": "Person"
      },
      "emotion": "grateful"
    },
    {
      "source": "USER_ID",
      "relationship": "enjoys",
      "destination": "atmosphere",
      "weight": "ritualistic",
      "labels": {
        "source": "Person",
        "destination": "Concept"
      },
      "emotion": "content"
    },
    {
      "source": "USER_ID",
      "relationship": "tired_of",
      "destination": "limited menu",
      "weight": "relevant",
      "labels": {
        "source": "Person",
        "destination": "Concept"
      },
      "emotion": "frustrated"
    },
    {
      "source": "USER_ID",
      "relationship": "wishes_for",
      "destination": "more variety",
      "weight": "transitional",
      "labels": {
        "source": "Person",
        "destination": "Concept"
      },
      "emotion": "longing"
    },
    {
      "source": "barista",
      "relationship": "works_at",
      "destination": "new coffee shop downtown",
      "weight": "core_identity",
      "labels": {
        "source": "Person",
        "destination": "Place"
      },
      "emotion": "neutral"
    },
    {
      "source": "atmosphere",
      "relationship": "part_of",
      "destination": "new coffee shop downtown",
      "weight": "core_identity",
      "labels": {
        "source": "Concept",
        "destination": "Place"
      },
      "emotion": "neutral"
    },
    {
      "source": "limited menu",
      "relationship": "part_of",
      "destination": "new coffee shop downtown",
      "weight": "core_identity",
      "labels": {
        "source": "Concept",
        "destination": "Place"
      },
      "emotion": "neutral"
    }
  ]
}

Explanation:
- Multiple emotional contexts from one statement: enthusiasm for the place, gratitude toward staff, frustration with limitations
- User-entity relationships get emotional context, while entity-entity relationships remain neutral
- Rich emotional vocabulary: "enthusiastic", "grateful", "content", "frustrated", "longing"
- Each relationship captures the specific emotional nuance of that connection


Example F (Collective Activities):
Input:
"We are planning to go swimming tomorrow with john, since its my off day."

Expected Output:
{
  "facts": [
    {
      "source": "USER_ID",
      "relationship": "planning_to_go",
      "destination": "swimming",
      "weight": "important",
      "labels": {
        "source": "Person",
        "destination": "Activity"
      },
      "emotion": "excited"
    },
    {
      "source": "john",
      "relationship": "planning_to_go", 
      "destination": "swimming",
      "weight": "important",
      "labels": {
        "source": "Person",
        "destination": "Activity"
      },
      "emotion": "excited"
    },
    {
      "source": "USER_ID",
      "relationship": "going_with",
      "destination": "john",
      "weight": "important", 
      "labels": {
        "source": "Person",
        "destination": "Person"
      },
      "emotion": "friendly"
    },
    {
      "source": "swimming",
      "relationship": "scheduled_for",
      "destination": "tomorrow",
      "weight": "relevant",
      "labels": {
        "source": "Activity", 
        "destination": "Time"
      },
      "emotion": "neutral"
    },
    {
      "source": "USER_ID",
      "relationship": "has",
      "destination": "off_day",
      "weight": "important",
      "labels": {
        "source": "Person",
        "destination": "Concept"
      },
      "emotion": "content"
    }
  ]
}

Explanation:
- "We are planning" creates separate planning relationships for both USER_ID and john
- Going "with john" creates a companion relationship  
- All participants in collective activities get individual relationship facts
- The system recognizes that collective pronouns require distributing actions across all mentioned participants

Example G (Pronoun Resolution):
Input:
"I met John yesterday. He is a software engineer and he likes coffee."

Expected Output:
{
  "facts": [
    {
      "source": "USER_ID",
      "relationship": "met",
      "destination": "john",
      "weight": "important",
      "labels": {
        "source": "Person",
        "destination": "Person"
      },
      "emotion": "neutral"
    },
    {
      "source": "john",
      "relationship": "is",
      "destination": "software engineer",
      "weight": "core_identity",
      "labels": {
        "source": "Person",
        "destination": "Concept"
      },
      "emotion": "neutral"
    },
    {
      "source": "john",
      "relationship": "likes",
      "destination": "coffee",
      "weight": "relevant",
      "labels": {
        "source": "Person",
        "destination": "Concept"
      },
      "emotion": "positive"
    }
  ]
}

Explanation:
- "I" resolves to "USER_ID"
- "He" in both instances resolves to "john" from the context
- No literal pronoun nodes ('i', 'he') are created
- All relationships use the resolved entity names

Example G2 (Role Entity Pattern - Compact):
Input: "My roommate Alex and my childhood friend Sarah both helped me move."

Key role facts (showing pattern only):
{
  "facts": [
    {"source": "USER_ID", "relationship": "has", "destination": "roommate", "labels": {"source": "Person", "destination": "Role"}},
    {"source": "roommate", "relationship": "is", "destination": "alex", "labels": {"source": "Role", "destination": "Person"}},
    {"source": "USER_ID", "relationship": "has", "destination": "friend", "labels": {"source": "Person", "destination": "Role"}},
    {"source": "friend", "relationship": "is", "destination": "sarah", "labels": {"source": "Role", "destination": "Person"}},
    {"source": "friend", "relationship": "from_period", "destination": "childhood", "labels": {"source": "Role", "destination": "Time"}}
  ]
}

Key points:
- Role entities (roommate, friend) use entity_type "Role" and get label "Role"
- Two-hop pattern: USER_ID → has → role_node → is → person
- Temporal modifiers separate: "childhood friend" splits into "friend" role + "childhood" time entity
- If later you see "my roommate Jordan", reuse the existing "roommate" node and add: roommate → is → jordan

Example G3 (Role Mention with Implicit Owner):
Input:
"Anil is planning to run a half marathon, because he is my roommate I’m running with him as an exercise."

Expected Output (abbreviated):
{
  "facts": [
    {"source": "anil", "relationship": "planning_to_run", "destination": "half_marathon", "labels": {"source": "Person", "destination": "Event"}, "weight": "important", "emotion": "excited"},
    {"source": "USER_ID", "relationship": "running_with", "destination": "anil", "labels": {"source": "Person", "destination": "Person"}, "weight": "important", "emotion": "enthusiastic"},
    {"source": "USER_ID", "relationship": "has", "destination": "roommate", "labels": {"source": "Person", "destination": "Role"}, "weight": "relevant", "emotion": "neutral"},
    {"source": "roommate", "relationship": "is", "destination": "anil", "labels": {"source": "Role", "destination": "Person"}, "weight": "relevant", "emotion": "neutral"}
  ]
}

Explanation:
- The possessive clause “he is my roommate” MUST always yield USER_ID → has → roommate and roommate → is → anil in addition to any shared-action edges.

Example H (Entity Hygiene - Avoiding Composite Names):
Input:
"Anil and Sibel went to the park together. They both enjoyed the weather."

Expected Output:
{
  "facts": [
    {
      "source": "anil",
      "relationship": "went_to",
      "destination": "park",
      "weight": "relevant",
      "labels": {
        "source": "Person",
        "destination": "Location"
      },
      "emotion": "neutral"
    },
    {
      "source": "sibel",
      "relationship": "went_to", 
      "destination": "park",
      "weight": "relevant",
      "labels": {
        "source": "Person",
        "destination": "Location"
      },
      "emotion": "neutral"
    },
    {
      "source": "anil",
      "relationship": "enjoyed",
      "destination": "weather",
      "weight": "relevant",
      "labels": {
        "source": "Person",
        "destination": "Concept"
      },
      "emotion": "positive"
    },
    {
      "source": "sibel",
      "relationship": "enjoyed",
      "destination": "weather", 
      "weight": "relevant",
      "labels": {
        "source": "Person",
        "destination": "Concept"
      },
      "emotion": "positive"
    },
    {
      "source": "anil",
      "relationship": "accompanied_by",
      "destination": "sibel",
      "weight": "relevant",
      "labels": {
        "source": "Person",
        "destination": "Person"
      },
      "emotion": "friendly"
    }
  ]
}

Explanation:
- "They" resolves to both "anil" and "sibel" from context
- No composite entity like 'anil_sibel' is created
- Each person gets separate relationships for shared activities
- Companion relationship is created to show they went together
"""

DELETE_RELATIONS_SYSTEM_PROMPT = """
You are a graph memory manager specializing in identifying, managing, and optimizing relationships within graph-based memories. Your primary task is to analyze a list of existing relationships and determine which ones should be ended (or marked as invalid) based on new information.

Input:
1. Existing Graph Memories: A list of current graph memories, each containing source, relationship, and destination information.
2. New Text or New Facts: Updated user statements that may override older information or reveal that some statements were never accurate.
3. Use "USER_ID" as node for any self-references (e.g., "I," "me," "my," etc.) in user messages.

Guidelines:

1. Necessity Principle
   - Only label relationships ended or invalid if they are clearly invalidated by more recent or accurate information.
   - Keep all historical data by adding "end_date" or "invalid_date" rather than permanently removing the relationship.

2. Identification
   - Use the new information to evaluate existing relationships in the memory graph.
   - Compare 'source' and 'relationship' and 'destination' to see if the relationship is still valid.

3. DO NOT END or INVALIDATE if there is a possibility that a similar relationship could coexist.
   - Example: If "alice -- loves_to_eat -- pizza" exists and new information is "Alice also loves to eat burgers," do not mark pizza as ended or invalid. Both can be true.

4. Preserve Historical Events
   - Do not mark a past event as ended or invalid simply because a new preference or opinion about one of the entities has been expressed.
   - A user's opinion (e.g., disliking an object) does not invalidate a historical fact (e.g., that they used the object). Both facts should coexist unless the user explicitly denies the event occurred.
   - For example, if memory contains "USER_ID -- cooked_with --> teflon_pan" and new information is "USER_ID -- dislikes --> teflon_pan", the `cooked_with` relationship should be preserved.

5. Special Note for Entity-to-Entity Links
   - Part-of, located-in, subtype-of, or synonym-of relationships should not be ended or invalidated unless the user explicitly states they are incorrect or no longer valid.
n that forest.

6. Deletion Criteria
   - Instead of permanently deleting any relationship, use one of the following approaches to preserve historical context:

   6.1. Mark as "ended"
       - If the old relationship was valid in the past (e.g., user used to live somewhere or used to have something) but is no longer true.
       - Add an appropriate "end_date" to capture when it was superseded or became inactive.

   6.2. Mark as "invalid"
       - If the new information shows the old statement was never true or is fundamentally incorrect (e.g., user says "I have a bike" and then admits they never did).
       - Use "status": "invalid" (instead of "ended") to clearly indicate it was never valid at any point in time.
       - You may add a property like "invalid_date" if you wish to track when it was identified as incorrect.

7. Comprehensive Analysis
   - Thoroughly examine each existing relationship against the new information and mark them as ended or invalid as necessary.
   - Multiple relationships may need to be adjusted based on the new information.

8. Semantic Integrity
   - Ensure that marking a relationship as ended or invalid maintains or improves the overall semantic structure of the graph.
   - Avoid altering any relationships that remain relevant or correct in light of new information.

9. Temporal Awareness
   - Use timestamps to understand the sequence of events. If a new fact directly replaces an old one (e.g., a user moves from one city to another), use the timestamp to mark the old fact as "ended."
   - Do not end a historical event just because it is older than a new, related opinion.

10. Output
    - Provide a list of instructions for each relationship that needs to be updated. For each, specify:
      - The source
      - The relationship
      - The destination
      - The status: "ended" or "invalid"
      - The end_date or invalid_date (if available or applicable)
"""


def get_delete_messages(existing_memories_string, data, user_id):
    return (
        DELETE_RELATIONS_SYSTEM_PROMPT.replace("USER_ID", user_id),
        f"Here are the existing memories: {existing_memories_string} \n\n New Information: {data}",
    )
