# User Reference Resolution Plan

## Problem Description
The app uses `mem0` to capture user-supplied information as both vector embeddings and graph edges so that future conversations with the LLM feel personal and contextual. During retrieval, `mem0.search` pulls relevant memories from vector and graph stores, and those memories feed a response generator that carries on the conversation as if it “knows” the user.  
The current gap is around how people mention other individuals. Users often refer to someone by a relationship title or role instead of a real name (e.g., “my roommate’s motorcycle,” “coach was sick”). Many users have multiple roommates, friends, or coaches, which makes it unclear which existing person node in the graph should receive the new memory. Misattributing the update weakens later retrieval and may corrupt the knowledge graph.

## Desired Solution
- Normalize graph inserts so people and objects always resolve to their canonical nodes before we attach new memories.
- Resolve references like “roommate” or “coach” to the correct person node before writing new memories.
- Use existing graph data (relationships, past mentions, attributes) and vector context to infer the most likely individual automatically.
- When multiple candidates are plausible, fall back to collecting a clarification from the user with minimal friction.
- Ensure that once the correct individual is known, all new data is stored under their canonical node so that future context pulls remain accurate.

## Target Graph Structure
- Create the main user's canonical `person` node once (e.g., `mustafa`) during initialization; every subsequent fact must reference this same node rather than recreating it.
- Person nodes represent real people (e.g., `anil`, `akin`, `mustafa`).
- The main user must be stored as a `person`-labeled node as well; all people (user or contacts) live in the same canonical namespace.
- Role/relationship nodes (e.g., `roommate`, `coach`) are shared descriptors; multiple people can point to the same role.
- Ownership/relationship edges capture who fills each role (`anil -is-> roommate`, `mustafa -has-> roommate`).
- Object nodes describe concrete items or concepts, and attribute edges break multi-word mentions into smaller facts (`anil -bought-> motorcycle`, `motorcycle -is-> second hand`).
- Attribute nodes must also anchor back to the relevant object or location (`menu -is-> limited`, `menu -belongs_to-> new_coffee_shop_downtown`) so later queries can traverse from the property to its owner.
- Each new fact expands the existing subgraph rather than creating duplicate “role” nodes, so later lookups have richer context to disambiguate references.
- Person nodes also carry additional metadata. Current relationship metadata persisted by `mem0.add` includes: `weight`, `is_uncertain`, `status`, `start_date`, `end_date`, `emotion`, `last_mentioned`, and `usage_count`. Preserve and propagate these when normalizing facts.

### Example Flow
1. Existing graph already stores:
   - `anil -is-> roommate`
   - `mustafa -has-> roommate`
2. User says: “My roommate bought a second-hand motorcycle.”
   - Resolve “my roommate” to the person node connected to `mustafa` through the shared `roommate` descriptor (prefer `anil` if it is the only match).
   - Insert normalized edges: `anil -bought-> motorcycle`, `motorcycle -is-> second hand`.
3. Later input: “Anil’s brother Akin is moving in with us; he is my roommate now.”
   - Create/update `akin` person node; connect `akin -is-> roommate`, ensure `mustafa -has-> roommate` already exists.
4. Subsequent message: “My roommate brought us food from work.”
   - Two candidates fill the `roommate` role (`anil`, `akin`).
   - Use neighboring facts (`akin -works-> swiss hotel`, `akin -is-> chef`, `anil -works-> sungrow`) to infer who “brought food” likely refers to (bias toward `akin` because culinary context matches).
   - If confidence remains low, ask the user which roommate they mean before writing the new fact (`akin -brought-> meal`).

This schema ensures role descriptors stay global, person nodes remain canonical, and attribute-rich edges support better reference resolution over time.

## Constraints & Considerations
- Titles can be nested within sentences (“my roommate’s motorcycle broke”) or stand alone (“coach canceled practice”), so extraction must work for varied phrasing.
- The same role may belong to multiple people simultaneously; ambiguity should trigger clarification rather than the wrong attribution.
- Some users may mention new individuals that are not yet stored; the system must gracefully create new nodes when no match exists.
- The solution must integrate cleanly with the existing `mem0` write pipeline so it does not break current memory ingestion and retrieval.
- Clarification prompts must reuse the existing LLM chat response channel; no separate UX or tooling is planned.
- `mem0.add` already handles entity extraction and graph mutations, so any clarification logic must hook in without duplicating that work.

## Proposed Approach
1. Extract entities, roles, and objects from each user message; normalize them to canonical node IDs when possible.
2. Detect when a mention refers to a role/title instead of a known name (NER or prompt-based classification).
3. Traverse the graph to find people connected to the user through the shared role node; include attribute-based filters (workplace, hobbies, prior actions) to refine candidates.
4. Score candidates using conversation context (recent chat history, vector similarity, relationship metadata) and normalized graph edges.
5. When multiple candidates remain:
   - Stop the `mem0.add` pipeline before writing any new graph edges.
   - Generate a short clarification question (“Are you talking about Anil or Akin?”) using the LLM response flow; no new UX is required.
   - Cache the ambiguous role candidates for the current session so the follow-up answer can fill in the missing data quickly.
6. Once the user clarifies, merge the triggering input and the clarification into a resolved statement (e.g., `“Anil didn’t flush the toilet; I told Anil before”`) and re-run `mem0.add` so the correct person node receives the update.
7. Insert new edges using the normalized schema (`person -> action -> object`, `object -> attribute -> value`) before writing to the graph/vector stores.
8. Log unresolved cases for future tuning of heuristics or prompts, and record clarifications so repeated questions are minimized within the same session.

## Decisions & Confirmations
- Database will be reset before rollout, so no legacy migration work is required; just validate the normalized schema on a clean store.
- Clarification questions ride through the existing LLM reply flow—no separate UX components.
- The primary user should have a single `person` node, consistent with every other person mentioned, but current prompts sometimes fail to create it correctly; address this in the extraction prompt or post-processing.

## Open Questions & Assumptions
- Session-level caching for role-to-person resolutions is still under evaluation (balance server load vs. correctness).
- After the reset, confirm the pipeline (including extraction prompts) reliably labels the main user as `person`; adjust prompts if gaps persist.

## Task Backlog (fill in as work progresses)
- [ ] Define the normalized graph schema (person, role, object nodes) and update ingestion code.
- [ ] Define the data structures and endpoints needed to look up relationship-based references.
- [ ] Implement the role/title detection layer and document prompt/logic.
- [ ] Build the candidate scoring logic and confidence thresholding leveraging normalized neighbors.
- [ ] Integrate the pause/clarify/resume flow into `mem0.add` so ambiguous inserts ask the user and replay the resolved message.
- [ ] Evaluate optional session-level caching for role-to-person resolutions and document the decision.
- [ ] Add tests or traces that validate correct attribution and log edge cases.
- [ ] Rewrite prompt/examples (e.g., `mem0/graphs/utils.py`) to reflect the normalized node structure, shared role descriptors, and attribute linking.

## Notes for Future Sessions
- Record any new heuristics or prompts that improve resolution accuracy.
- Track common roles/titles that still fail so we can expand the detection vocabulary.
- Use this file to capture decisions, progress, and remaining questions so you do not need to re-explain the problem each session.
- Capture how ambiguous-role clarification caching performs (latency savings, accuracy impacts) once implemented.
- Confirm new graph writes label the main user as `person` after the database reset.
