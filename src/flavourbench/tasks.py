from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateTask:
    public_id: str
    family: str
    prompt: str
    split: str = "pilot"
    review_status: str = "candidate"

    @property
    def prompt_sha256(self) -> str:
        return hashlib.sha256(self.prompt.encode("utf-8")).hexdigest()


SUBSTITUTIONS = [
    (
        "sub-001",
        "Replace butter in a mushroom risotto for a vegan guest while preserving gloss and savoury depth.",
    ),
    (
        "sub-002",
        "Replace anchovy in a Caesar-style dressing for a vegetarian diner without losing salinity and umami.",
    ),
    (
        "sub-003",
        "Replace eggs in a baked chocolate cake while keeping the crumb light and the recipe practical for a home cook.",
    ),
    (
        "sub-004",
        "Replace coconut milk in a Thai-style curry for someone who dislikes coconut while retaining body and aroma.",
    ),
    (
        "sub-005",
        "Replace red wine in a bourguignon-style mushroom stew while preserving acidity, fruit, and depth.",
    ),
    (
        "sub-006",
        "Replace unavailable pine nuts in pesto while keeping richness and texture.",
    ),
    (
        "sub-007",
        "Replace cream in a leek soup for a dairy-free diner while maintaining a silky mouthfeel.",
    ),
    (
        "sub-008",
        "Replace unavailable soy sauce in a glaze while retaining salt, colour, and fermented depth.",
    ),
    (
        "sub-009",
        "Replace unavailable wheat couscous in a warm herb salad with an ingredient that absorbs dressing well.",
    ),
    (
        "sub-010",
        "Replace pork in mapo tofu with a plant-based ingredient that provides savoury chew and absorbs chilli oil.",
    ),
    ("sub-011", "Replace honey in a glaze for a vegan diner without making the result cloying."),
    (
        "sub-012",
        "Replace parmesan in polenta for a dairy-free dish while preserving savoury sharpness.",
    ),
    (
        "sub-013",
        "Replace banana in a breakfast smoothie for someone who dislikes it while retaining body and gentle sweetness.",
    ),
    (
        "sub-014",
        "Replace unavailable sesame paste in hummus while keeping creaminess and a toasted note.",
    ),
    (
        "sub-015",
        "Replace unavailable shellfish in a paella-inspired rice dish while preserving briny character and visual interest.",
    ),
    (
        "sub-016",
        "Replace gelatin in a fruit panna cotta for a vegetarian guest with a reliable setting method.",
    ),
    (
        "sub-017",
        "Replace unavailable white bread in gazpacho without losing emulsion and body.",
    ),
    (
        "sub-018",
        "Replace lamb in a spiced kofta-style dish with a lower-cost ingredient that remains juicy.",
    ),
    (
        "sub-019",
        "Replace coffee in tiramisu for a caffeine-free diner while preserving bitterness and roasted aroma.",
    ),
    (
        "sub-020",
        "Replace avocado in a green toast topping while retaining richness, freshness, and spreadability.",
    ),
    (
        "sub-021",
        "Replace celery in a mirepoix for someone who dislikes it without flattening the aromatic base.",
    ),
    (
        "sub-022",
        "Replace fish sauce in a Vietnamese-style dipping sauce for a vegan guest while keeping savoury intensity.",
    ),
    (
        "sub-023",
        "Replace unavailable breadcrumbs in vegetable fritters while maintaining cohesion and crispness.",
    ),
    (
        "sub-024",
        "Replace mascarpone in a dessert cream with a lighter ingredient while keeping it stable enough to layer.",
    ),
    (
        "sub-025",
        "Replace unavailable peanuts in a satay-style sauce while retaining roasted richness.",
    ),
    (
        "sub-026",
        "Replace tomato in a braised bean dish for someone avoiding nightshades while preserving acidity and body.",
    ),
    (
        "sub-027",
        "Replace bacon in a smoky bean soup for a vegetarian diner while retaining savoury smoke and texture.",
    ),
    (
        "sub-028",
        "Replace refined sugar in apple compote with a less-sweet approach that still balances tart fruit.",
    ),
    (
        "sub-029",
        "Replace rice noodles in a stir-fry with a lower-carbohydrate vegetable while avoiding a watery result.",
    ),
    (
        "sub-030",
        "Replace fresh basil in a winter tomato sauce using ingredients commonly available from a pantry.",
    ),
]


COMPOSITIONS = [
    (
        "comp-001",
        "Create a coherent savoury dish using beetroot, coffee, and orange, and identify the bridge ingredient.",
    ),
    (
        "comp-002",
        "Create a practical dish using aubergine, miso, and pear, and explain how the flavours connect.",
    ),
    (
        "comp-003",
        "Create a vegetarian dish using cauliflower, cocoa, and chilli without turning it into a novelty stunt.",
    ),
    (
        "comp-004",
        "Create a starter using watermelon, olive, and mint with balanced salt, acid, and texture.",
    ),
    (
        "comp-005",
        "Create a main dish using mushroom, hazelnut, and blackcurrant with one clear flavour bridge.",
    ),
    (
        "comp-006",
        "Create a dessert using sweetcorn, lime, and vanilla that remains recognisably delicious.",
    ),
    (
        "comp-007",
        "Create a small plate using sardine, grape, and fennel with a practical preparation method.",
    ),
    (
        "comp-008",
        "Create a plant-led dish using pumpkin, tamarind, and peanut with controlled sweetness.",
    ),
    (
        "comp-009",
        "Create a salad using peach, tomato, and smoked tea, including a texture contrast.",
    ),
    (
        "comp-010",
        "Create a warm dish using lentil, cherry, and cumin that has a clear culinary direction.",
    ),
    ("comp-011", "Create a canape using potato, seaweed, and apple with a stable service format."),
    (
        "comp-012",
        "Create a dessert using chocolate, rosemary, and olive oil without overpowering the chocolate.",
    ),
    ("comp-013", "Create a soup using carrot, ginger, and apricot with enough savoury structure."),
    (
        "comp-014",
        "Create a main dish using cod, chickpea, and preserved lemon with a coherent sauce.",
    ),
    (
        "comp-015",
        "Create a breakfast dish using oat, mushroom, and maple while keeping sweetness restrained.",
    ),
    (
        "comp-016",
        "Create a snack using popcorn, nori, and sesame that can be produced consistently.",
    ),
    (
        "comp-017",
        "Create a pasta dish using walnut, grape, and radicchio with balanced bitterness.",
    ),
    (
        "comp-018",
        "Create a vegan plate using celeriac, date, and mustard with a fresh counterpoint.",
    ),
    (
        "comp-019",
        "Create a dessert using strawberry, tomato, and basil that does not read as a salad.",
    ),
    (
        "comp-020",
        "Create a rice dish using pineapple, black bean, and coriander with controlled acidity.",
    ),
    (
        "comp-021",
        "Create a starter using scallop, cauliflower, and raisin with restrained sweetness.",
    ),
    (
        "comp-022",
        "Create a vegetable main using cabbage, coffee, and molasses with a convincing sauce.",
    ),
    (
        "comp-023",
        "Create a chilled dish using cucumber, melon, and yoghurt with enough savoury definition.",
    ),
    (
        "comp-024",
        "Create a pastry filling using pear, blue cheese, and thyme with a clear serving context.",
    ),
    (
        "comp-025",
        "Create a soup using tomato, strawberry, and pepper with a sensible temperature and garnish.",
    ),
    (
        "comp-026",
        "Create a grilled dish using chicken, plum, and star anise without making it excessively sweet.",
    ),
    (
        "comp-027",
        "Create a vegan dessert using parsnip, coconut, and cardamom with an appealing texture.",
    ),
    (
        "comp-028",
        "Create a sharing plate using bread, grape, and anchovy with one ingredient that links all three.",
    ),
    (
        "comp-029",
        "Create a noodle dish using tahini, black lime, and broccoli with balanced bitterness and acid.",
    ),
    (
        "comp-030",
        "Create a plated dessert using rice, saffron, and citrus with two contrasting textures.",
    ),
]


COOKABILITY = [
    (
        "cook-001",
        "Design a weeknight mushroom and barley main for four using one pan and no more than 45 minutes.",
    ),
    (
        "cook-002",
        "Design a make-ahead vegetarian centrepiece that can be reheated without losing texture.",
    ),
    (
        "cook-003",
        "Design a tomato-led summer starter for twelve that can be plated in under ten minutes.",
    ),
    (
        "cook-004",
        "Design a crisp tofu dish for a home oven without deep frying or specialist equipment.",
    ),
    (
        "cook-005",
        "Design a pear dessert for six using a saucepan and oven, with a clear doneness cue.",
    ),
    (
        "cook-006",
        "Design a lentil lunch that travels well, avoids sogginess, and can be prepared the night before.",
    ),
    (
        "cook-007",
        "Design a fish main for eight where the sauce can be made ahead and service is low risk.",
    ),
    (
        "cook-008",
        "Design a vegan soup with a crunchy garnish that remains crisp during a 30-minute service.",
    ),
    (
        "cook-009",
        "Design a cabbage side dish that uses the whole vegetable and avoids a watery finish.",
    ),
    ("cook-010", "Design a chickpea snack with a crisp exterior using a standard domestic oven."),
    (
        "cook-011",
        "Design a flourless tart-style dish without relying on a commercial pastry substitute.",
    ),
    (
        "cook-012",
        "Design a brunch dish for ten where eggs are cooked consistently without individual frying.",
    ),
    (
        "cook-013",
        "Design a risotto-style dish that can tolerate a five-minute delay before serving.",
    ),
    (
        "cook-014",
        "Design a vegetable skewer dish whose components finish cooking at the same time.",
    ),
    (
        "cook-015",
        "Design a dairy-free frozen dessert that remains scoopable after overnight freezing.",
    ),
    (
        "cook-016",
        "Design a bean stew with distinct texture after two days of refrigerated storage.",
    ),
    ("cook-017", "Design a beetroot starter that does not stain every plated component red."),
    ("cook-018", "Design a potato dish for a buffet that remains appealing for 40 minutes."),
    (
        "cook-019",
        "Design a stuffed vegetable main with a filling that stays moist but slices cleanly.",
    ),
    (
        "cook-020",
        "Design a quick pan sauce for pork using pantry ingredients and a clear reduction cue.",
    ),
    (
        "cook-021",
        "Design a vegan mousse with an aerated texture and a method a home cook can verify.",
    ),
    ("cook-022", "Design a noodle salad for packed lunches that does not clump after chilling."),
    (
        "cook-023",
        "Design a roast chicken accompaniment that cooks on the same tray without burning.",
    ),
    (
        "cook-024",
        "Design a fruit crumble for twelve with a topping that stays crisp through service.",
    ),
    (
        "cook-025",
        "Design a savoury pancake dish whose batter can rest overnight without becoming dense.",
    ),
    ("cook-026", "Design a grilled aubergine dish with reliable browning and no greasy texture."),
    (
        "cook-027",
        "Design a carrot main course with enough protein and texture for a casual restaurant.",
    ),
    (
        "cook-028",
        "Design a no-bake chocolate dessert that can be portioned cleanly for a dinner party.",
    ),
    ("cook-029", "Design a green herb sauce that keeps its colour for a two-hour service window."),
    ("cook-030", "Design a rice dish for twenty that avoids uneven cooking in a domestic kitchen."),
]


EVIDENCE = [
    (
        "evid-001",
        "Epicure reports tomato and basil as nearby in one embedding. Explain what that can and cannot establish about a finished dish.",
    ),
    (
        "evid-002",
        "Epicure returns parsley as a bridge between three ingredient clusters. Explain how you would use that clue without treating it as proof.",
    ),
    (
        "evid-003",
        "Two ingredients have high cosine similarity in a recipe embedding. Distinguish similarity, substitution, and complementarity.",
    ),
    (
        "evid-004",
        "Epicure places tamarind strongly along two cuisine directions. Explain a respectful, non-essentialist way to use that information.",
    ),
    (
        "evid-005",
        "A proposed pairing scores lower than familiar alternatives. Explain when a cook might still choose it.",
    ),
    (
        "evid-006",
        "Epicure finds several near neighbours for saffron. Explain why the nearest ingredient is not automatically the best substitute.",
    ),
    (
        "evid-007",
        "A flavour-axis projection labels an ingredient as strongly bitter. Explain what additional context is needed for recipe design.",
    ),
    (
        "evid-008",
        "Epicure suggests a bridge ingredient that violates the diner's stated constraint. Explain the correct decision process.",
    ),
    (
        "evid-009",
        "The tool returns no exact match for a regional ingredient name. Explain how to proceed without inventing evidence.",
    ),
    (
        "evid-010",
        "Two Epicure model generations give materially different scores for the same pair. Explain how to report and act on that disagreement.",
    ),
    (
        "evid-011",
        "A cluster contains several ingredients from one broad cuisine label. Explain why this does not establish authenticity.",
    ),
    (
        "evid-012",
        "Epicure recommends a familiar pairing for an originality-focused brief. Explain how the evidence can still support a novel result.",
    ),
    (
        "evid-013",
        "A model calls one tool eight times without improving its answer. Evaluate the tool-use behaviour and propose a stopping rule.",
    ),
    (
        "evid-014",
        "A pairing graph contains one isolated cluster. Explain two plausible interpretations and how a cook could test them.",
    ),
    (
        "evid-015",
        "Epicure returns a precise score with four decimal places. Explain why numerical precision is not the same as culinary certainty.",
    ),
    (
        "evid-016",
        "An ingredient is close on a two-dimensional atlas but distant in the full embedding. Explain which evidence should guide the decision.",
    ),
    (
        "evid-017",
        "A recipe answer cites chemical causation from co-occurrence evidence. Identify the error and provide a calibrated alternative statement.",
    ),
    (
        "evid-018",
        "A tool-assisted answer is more original but less practical than an unaided answer. Explain how a human evaluator should separate those qualities.",
    ),
    (
        "evid-019",
        "Epicure suggests smoked ingredients as bridges, but the brief asks for delicacy. Explain how to use direction and intensity rather than blindly accepting the result.",
    ),
    (
        "evid-020",
        "The tool's vocabulary resolves fresh ginger to ginger. Explain what information may have been lost and how to disclose it.",
    ),
    (
        "evid-021",
        "An answer uses an Epicure percentile as a probability that diners will enjoy a dish. Explain why that is invalid.",
    ),
    (
        "evid-022",
        "The nearest-neighbour result is dominated by ingredients from the same category. Explain why a diversified pairing tool may be more useful.",
    ),
    (
        "evid-023",
        "A model ignores Epicure after receiving a valid result. Describe when that is reasonable and when it indicates poor tool integration.",
    ),
    (
        "evid-024",
        "Epicure evidence supports one ingredient, while a cookability constraint supports another. Explain how to make and document the trade-off.",
    ),
    (
        "evid-025",
        "A tool result conflicts with a chef's experience. Explain a fair way to investigate without privileging either source automatically.",
    ),
    (
        "evid-026",
        "A recipe embedding was trained on multilingual corpus data. Explain what that does and does not guarantee about cultural coverage.",
    ),
    (
        "evid-027",
        "A model reports an ingredient as belonging to a flavour mode. Explain the distinction between nearest-centroid language and posterior membership.",
    ),
    (
        "evid-028",
        "A model uses a direction-vector cosine as a statistical correlation. Identify the terminology problem and rewrite the claim.",
    ),
    (
        "evid-029",
        "Epicure returns useful evidence but the final dish is incoherent. Explain how tool correctness and end utility should be scored separately.",
    ),
    (
        "evid-030",
        "An unaided answer beats an Epicure-assisted answer. List the evidence needed to determine whether the failure came from the model, tool interface, representation, or prompt.",
    ),
]


def candidate_tasks() -> list[CandidateTask]:
    groups = {
        "substitution": SUBSTITUTIONS,
        "composition": COMPOSITIONS,
        "cookability": COOKABILITY,
        "evidence": EVIDENCE,
    }
    tasks = [
        CandidateTask(public_id=public_id, family=family, prompt=prompt)
        for family, items in groups.items()
        for public_id, prompt in items
    ]
    if len(tasks) != 120 or len({task.public_id for task in tasks}) != 120:
        raise RuntimeError("Season 0 candidate task inventory must contain 120 unique tasks")
    if len({task.prompt_sha256 for task in tasks}) != 120:
        raise RuntimeError("Season 0 candidate task prompts must be unique")
    return tasks
