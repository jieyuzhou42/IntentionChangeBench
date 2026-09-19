"""Final-selection policy shared by evaluation entry points."""

BEST_AVAILABLE_RULES = """Final action policy (best available concrete choice):
- Submit your best concrete available solution even when no option satisfies all
  user conditions. Make the trade-off yourself; do not defer it to the user.
- Use your own inferred must_have/preferred/optional priorities to compare whole
  solutions. Protect must_have first, then preferred, then optional. If user
  must-have requirements conflict, choose the best achievable compromise and
  identify exactly which requirements the selected solution does not meet.
- Do not change the intention prediction to make your chosen action look compliant.
  Report unmet conditions in the action rationale; a disclosed violation is still
  a violation for evaluation.
- Select actual supported options. Do not return an unchosen list of alternatives,
  'such as', 'to be confirmed', 'select later', or 'if available' as the final choice.
- Continue researching alternatives when tools are available: a missing flight can
  require searching a ground route; an unavailable room type can require another
  room type. Do not stop at the first failed exact match.
- Never invent candidates, availability, prices or attributes. Resource restrictions
  such as minimum hotel nights and occupancy are feasibility conditions: choose a
  different supported option or adjust the stay and explicitly disclose the user
  requirement being sacrificed. Do not present an impossible booking as feasible.
- Only a genuine absence of any supported alternative after the available research
  may be reported as infeasible. This is an explicit failure, not an unresolved
  placeholder or a successful plan. A lack of a perfect match is not this exception.
""".strip()

TRAVEL_SELECTION_RULES = BEST_AVAILABLE_RULES + """
- In the final itinerary, commit to a named accommodation for every overnight stay,
  a specific supported transport option for each required journey (including the
  trip endpoint/return), and concrete meal choices. Self-catering or packed food
  can be a concrete choice when explicitly selected, not a promise to decide later.
- Write a complete daily itinerary with day, current_city, transportation,
  breakfast, lunch, dinner, attraction and accommodation. Use '-' only when an
  item is intentionally not needed, such as lodging after the trip ends or an
  intentionally omitted attraction; explain sacrificed requirements in rationale.
- Account for accommodation per occupied/billed night, fares for all travelers,
  and meal costs for the party. Report known total costs and genuinely unpriced
  components without treating them as free or claiming a verified total budget.
"""
