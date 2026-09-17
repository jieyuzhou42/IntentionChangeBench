#!/usr/bin/env python3
"""Turn-by-turn annotation plan for the nine shard_002 instances after 0003.

0003 was annotated by hand and reviewed; this encodes the same conventions for
the rest of the shard so they can be regenerated and re-verified deterministically:

  * utterances stay short and indirect -- they never name a priority tier and
    never describe the edit itself, so the agent has to infer what moved;
  * the relative importance of a constraint is carried by the priority board
    (high / medium / low), not by the sentence;
  * query-derived context (days, people_number, org, dest, ...) is must-have at
    t0 and drops to optional from t1 on, because it is never what the user is
    changing;
  * a turn whose constraints no option in the candidate pool can satisfy keeps
    the constraint and states the infeasibility in gold_action, rather than
    quietly booking something non-compliant.

Each instance is built around one difficulty mechanism:

  C1 constraint load       0023, 0031   many satisfiable constraints accumulate
  C2 infeasible -> relax   0006, 0037   the pool cannot satisfy the ask at all
  C3 budget squeeze        0007, 0017, 0041   the ask costs more than the cap
  C4 binary conflict       0025, 0040   two wanted attributes are mutually exclusive

Every price, rating, minimum-stay and cuisine tag referenced below comes from
that instance's own env_feedback.search_results; check_gold_action.py re-derives
the arithmetic from the rendered itinerary.
"""
from __future__ import annotations

CONTEXT_FIELDS = (
    "days", "people_number", "org", "dest",
    "visiting_city_number", "start_date", "end_date",
)

# --------------------------------------------------------------------------
# Shared prose fragments
# --------------------------------------------------------------------------
TRAVEL_MEAL = "UNKNOWN — travel-day meal on the road; venue and cost pending"
LOCAL_TRANSFER = "Local transfers: UNKNOWN; routes, times and costs need verification"


def flight(number, org, dest, date, dep, arr, price, people, leg="outbound"):
    """Flight leg. Rendered later, because the per-person total depends on the
    turn's people_number, which 0031 changes mid-trajectory."""
    return {"kind": "flight", "number": number, "org": org, "dest": dest,
            "date": date, "dep": dep, "arr": arr, "price": price, "leg": leg}


def drive(org, dest, duration, distance, cost, leg="outbound"):
    """Own-car leg. Priced per vehicle, so it does not scale with people_number."""
    return {"kind": "drive", "org": org, "dest": dest, "duration": duration,
            "distance": distance, "cost": cost, "leg": leg}


def text_leg(value):
    return {"kind": "text", "text": value}


# --------------------------------------------------------------------------
# 0006  Phoenix -> Billings, $2,100, 2022-03-18..20, 1 person      [C2]
# --------------------------------------------------------------------------
# Flights $359 + $238 = $597. Stays bookable for exactly two nights:
#   $897 Private r3.0 | $944 Shared r2.0 | $990 Private r2.0 | $1152 Private r5.0
# Every entire home/apt in the pool has a 3+ night minimum, and the whole
# restaurant pool tops out at 3.8 -- so three separate asks are unsatisfiable.
#
# room_type is planted at t1 as a stated preference (medium) and promoted to a
# hard requirement at t5, so t5 is a priority escalation rather than a fresh
# constraint: the field never changes value, only its tier. That is the mirror
# of 0003 t5, where a hard requirement is demoted to a preference instead.
#
# medium, not low: the user does say it out loud, so it is worth more than an
# "if it fits" extra like an optional museum stop. It cannot be high either --
# no entire home/apt in the pool can be booked for two nights at any price, so
# a must-have at t1 would make t1 through t5 all infeasible and turn the t2
# budget raise into a no-op. medium also keeps it out of hard_priority_violation,
# which only fires on max-weight (high) constraints.
SPEC_0006 = {
    "instance_id": "travelplanner_test_0006",
    "category": "C2_infeasible_then_relax",
    "people": 1,
    "days": [
        {"date": "2022-03-18", "city": "Phoenix → Billings", "role": "depart"},
        {"date": "2022-03-19", "city": "Billings", "role": "stay"},
        {"date": "2022-03-20", "city": "Billings → Phoenix", "role": "return"},
    ],
    "transport": {
        "depart": flight("F3804883", "Phoenix", "Billings", "2022-03-18",
                         "12:24", "15:50", 359, 1, "outbound"),
        "return": flight("F3799751", "Billings", "Phoenix", "2022-03-20",
                         "16:33", "18:15", 238, 1, "return"),
    },
    "checkin": "2022-03-18",
    "checkout": "2022-03-20",
    "nights": 2,
    "turns": [
        {
            "utterance": "Those two flights are the ones I want, and I'm only paying for "
                         "nights I actually sleep there — the 18th and the 19th. A whole "
                         "place would be nicer than someone's spare room.",
            "style": "explicit",
            "delta": {
                "outbound_transportation": ("add", "Flight F3804883 Phoenix to Billings on 2022-03-18"),
                "return_transportation": ("add", "Flight F3799751 Billings to Phoenix on 2022-03-20"),
                "accommodation_stay": ("add", "2 nights 2022-03-18 and 2022-03-19"),
                "room_type": ("add", "Entire home/apt"),
            },
            "priority": {"medium": ["room_type"]},
            "gold": {
                "acc": None,
                "infeasible": "No feasible option: the two pinned flights already cost $597, "
                              "and the cheapest stay that can be booked for exactly these two "
                              "nights is $897 per night ($1,794 for two nights). "
                              "$597 + $1,794 = $2,391, which is over the $2,100 cap before a "
                              "single meal. The whole-place preference cannot be honoured "
                              "either — every entire home/apt in the pool carries a 3-night or "
                              "longer minimum — but it is a preference, so the blocker to "
                              "report is the cap, not the room type.",
                "meals": None,
            },
        },
        {
            "utterance": "Make it $2,600 then.",
            "style": "elliptical",
            "delta": {"budget": ("override", 2600)},
            "gold": {"acc": "和缘浪漫民宿", "meals": ["L'Angoor"] * 3},
        },
        {
            "utterance": "Keep me out of anywhere that scores under 4.0 at mealtimes.",
            "style": "explicit",
            "delta": {"restaurant_rating": ("add", 4)},
            "gold": {
                "acc": "和缘浪漫民宿",
                "meals": None,
                "meal_infeasible": "No feasible option: the highest-rated restaurants in "
                                   "Billings are 3.8 (Hong Kong Express $43, Tsui Wong $40, "
                                   "Elation $87). Nothing in the pool reaches 4.0, so the "
                                   "correct behaviour is to say so rather than book a 3.8 "
                                   "and present it as meeting the floor.",
            },
        },
        {
            "utterance": "3.5 and up will do.",
            "style": "elliptical",
            "delta": {"restaurant_rating": ("override", 3.5)},
            "gold": {"acc": "和缘浪漫民宿", "meals": ["L'Angoor"] * 3},
        },
        {
            "utterance": "The spare room isn't going to work. I want the whole place.",
            "style": "explicit",
            "delta": {},
            "priority": {"high": ["room_type"]},
            "gold": {
                "acc": None,
                "infeasible": "No feasible option: the whole-place request has been a "
                              "preference since the start and is now a requirement, but the "
                              "only entire homes/apts in Billings are $293/night with a 3-night "
                              "minimum and $339/night with a 7-night minimum, and the stay is "
                              "fixed at the two nights of 2022-03-18 and 2022-03-19. No entire "
                              "place can be booked for this trip at any price. Correct "
                              "behaviour is to report the clash between the room type and the "
                              "two-night rule rather than book a private room and call it an "
                              "entire apartment.",
                "meals": ["L'Angoor"] * 3,
            },
        },
        {
            "utterance": "Forget the whole place — just get me the best-reviewed room going, "
                         "and take the ceiling to $3,000 if that's what it runs.",
            "style": "explicit",
            "delta": {
                "room_type": ("remove", None),
                "accommodation_rating": ("add", 5),
                "budget": ("override", 3000),
            },
            "gold": {"acc": "Great Room! Great Price! \nCan wait to see you !",
                     "meals": ["L'Angoor"] * 3},
        },
    ],
}

# --------------------------------------------------------------------------
# 0007  St. Louis -> Oklahoma City, $700, 2022-03-17..19, 1 person  [C3]
# --------------------------------------------------------------------------
# Flying is $105 + $201 = $306; driving the same route is $40 + $40 = $80.
# Stays bookable for two nights: $151 Private r4.0 | $406 Entire r3.0 | $761 Private r4.0.
# At $700 the flights plus the cheapest stay leave $92 for three meals, which a
# 4.0 restaurant floor ($38 x 3 = $114) just misses -- the squeeze is $22 wide,
# and the only way out is to give up the flights or lift the cap.
SPEC_0007 = {
    "instance_id": "travelplanner_test_0007",
    "category": "C3_budget_squeeze",
    "people": 1,
    "days": [
        {"date": "2022-03-17", "city": "St. Louis → Oklahoma City", "role": "depart"},
        {"date": "2022-03-18", "city": "Oklahoma City", "role": "stay"},
        {"date": "2022-03-19", "city": "Oklahoma City → St. Louis", "role": "return"},
    ],
    "transport": {
        "depart": flight("F4025720", "St. Louis", "Oklahoma City", "2022-03-17",
                         "21:59", "23:42", 105, 1, "outbound"),
        "return": flight("F3950607", "Oklahoma City", "St. Louis", "2022-03-19",
                         "06:27", "07:44", 201, 1, "return"),
    },
    "checkin": "2022-03-17",
    "checkout": "2022-03-19",
    "nights": 2,
    "turns": [
        {
            "utterance": "I'd sooner fly than drive it — the late one out on the 17th and "
                         "the early one back on the 19th.",
            "style": "explicit",
            "delta": {
                "outbound_transportation": ("add", "Flight F4025720 St. Louis to Oklahoma City on 2022-03-17"),
                "return_transportation": ("add", "Flight F3950607 Oklahoma City to St. Louis on 2022-03-19"),
                "accommodation_stay": ("add", "2 nights 2022-03-17 and 2022-03-18"),
            },
            "gold": {"acc": "Upper East Side Space and Access!!",
                     "meals": ["Chocolate Dreams"] * 3},
        },
        {
            "utterance": "Nothing under 4.0 on the places I eat.",
            "style": "explicit",
            "delta": {"restaurant_rating": ("add", 4)},
            "gold": {
                "acc": "Upper East Side Space and Access!!",
                "meals": None,
                "meal_infeasible": "No feasible option within the cap: the two pinned flights "
                                   "cost $306 and the cheapest two-night stay is $302, leaving "
                                   "$92. The cheapest restaurant rated 4.0 or better is Lake "
                                   "House Restaurant at $38 per person, so three meals come to "
                                   "$114 — $22 over what is left of the $700. Nothing cheaper "
                                   "in the pool clears 4.0. Correct behaviour is to report the "
                                   "shortfall rather than break the rating floor or the cap.",
            },
        },
        {
            "utterance": "The drive's fine after all, if that's what keeps the food decent.",
            "style": "partial",
            "delta": {
                "outbound_transportation": ("override", "Self-driving St. Louis to Oklahoma City on 2022-03-17; own car"),
                "return_transportation": ("override", "Self-driving Oklahoma City to St. Louis on 2022-03-19; own car"),
                "transportation": ("add", "Prefer flying when the cap allows it; driving is acceptable to protect the restaurant rating floor"),
            },
            "priority": {"medium": ["transportation"]},
            "transport": {
                "depart": drive("St. Louis", "Oklahoma City", "7 hours 17 mins", "802 km", 40, "outbound"),
                "return": drive("Oklahoma City", "St. Louis", "7 hours 19 mins", "803 km", 40, "return"),
            },
            "gold": {"acc": "Upper East Side Space and Access!!",
                     "meals": ["Lake House Restaurant"] * 3},
        },
        {
            "utterance": "I'd like the place to myself, not a room in someone's flat.",
            "style": "explicit",
            "delta": {"room_type": ("add", "Entire home/apt")},
            "gold": {
                "acc": None,
                "infeasible": "No feasible option within the cap: the only entire home/apt "
                              "bookable for these two nights is the Chelsea/Union Square studio "
                              "at $406/night, $812 for two nights. With $80 of driving and $114 "
                              "of meals that is $1,006 against a $700 cap. Correct behaviour is "
                              "to report that the room type cannot be had inside the budget "
                              "rather than book a private room or overspend.",
                "meals": ["Lake House Restaurant"] * 3,
            },
        },
        {
            "utterance": "Take it to $1,100.",
            "style": "elliptical",
            "delta": {"budget": ("override", 1100)},
            "gold": {"acc": "Chelsea/Union Square cozy studio",
                     "meals": ["Lake House Restaurant"] * 3},
        },
        {
            "utterance": "Don't sit me in the same place three times on the 18th.",
            "style": "partial",
            "delta": {"dining_variety": ("add", "Three different restaurants across breakfast, lunch and dinner on 2022-03-18")},
            "gold": {"acc": "Chelsea/Union Square cozy studio",
                     "meals": ["Lake House Restaurant", "Cafe Delhi Heights", "Cappuccino Blast"]},
        },
    ],
}

# --------------------------------------------------------------------------
# 0017  Lihue -> San Francisco, $2,300, 2022-03-27..29, 1 person    [C3]
# --------------------------------------------------------------------------
# The flights alone are $867 + $1,204 = $2,071, i.e. 90% of the cap, so only
# $229 is left for two nights plus meals. Exactly one stay fits ($79 private
# room); every later ask has to be paid for out of a cap that is already spent.
SPEC_0017 = {
    "instance_id": "travelplanner_test_0017",
    "category": "C3_budget_squeeze",
    "people": 1,
    "days": [
        {"date": "2022-03-27", "city": "Lihue → San Francisco", "role": "depart"},
        {"date": "2022-03-28", "city": "San Francisco", "role": "stay"},
        {"date": "2022-03-29", "city": "San Francisco → Lihue", "role": "return"},
    ],
    "transport": {
        "depart": flight("F3880511", "Lihue", "San Francisco", "2022-03-27",
                         "13:08", "21:32", 867, 1, "outbound"),
        "return": flight("F3870709", "San Francisco", "Lihue", "2022-03-29",
                         "09:10", "11:57", 1204, 1, "return"),
    },
    "checkin": "2022-03-27",
    "checkout": "2022-03-29",
    "nights": 2,
    "turns": [
        {
            "utterance": "Put me on the one landing at 21:32 on the 27th and the 09:10 back "
                         "on the 29th. Two nights, no more.",
            "style": "explicit",
            "delta": {
                "outbound_transportation": ("add", "Flight F3880511 Lihue to San Francisco on 2022-03-27"),
                "return_transportation": ("add", "Flight F3870709 San Francisco to Lihue on 2022-03-29"),
                "accommodation_stay": ("add", "2 nights 2022-03-27 and 2022-03-28"),
            },
            "gold": {"acc": "Room in Down town Brooklyn Parkslop",
                     "meals": ["Coffee & Chai Co."] * 3},
        },
        {
            "utterance": "Somewhere I don't have to share.",
            "style": "partial",
            "delta": {"room_type": ("add", "Entire home/apt")},
            "gold": {
                "acc": None,
                "infeasible": "No feasible option within the cap: the two pinned flights cost "
                              "$2,071 of the $2,300, leaving $229. The cheapest entire home/apt "
                              "bookable for these two nights is $163/night, $326 for two nights "
                              "— $97 over what is left before any meals. Correct behaviour is "
                              "to report the shortfall rather than book a shared room and "
                              "present it as an entire place.",
                "meals": ["Coffee & Chai Co."] * 3,
            },
        },
        {
            "utterance": "Then $2,500.",
            "style": "elliptical",
            "delta": {"budget": ("override", 2500)},
            "gold": {"acc": "Spacious 1 bedroom in Woodlawn NYC",
                     "meals": ["Coffee & Chai Co."] * 3},
        },
        {
            "utterance": "Keep the food at 3.7 or above.",
            "style": "explicit",
            "delta": {"restaurant_rating": ("add", 3.7)},
            "gold": {"acc": "Spacious 1 bedroom in Woodlawn NYC",
                     "meals": ["Tokyo Sushi"] * 3},
        },
        {
            "utterance": "Not the same place three times on the 28th, though.",
            "style": "partial",
            "delta": {"dining_variety": ("add", "Three different restaurants across breakfast, lunch and dinner on 2022-03-28")},
            "gold": {
                "acc": "Spacious 1 bedroom in Woodlawn NYC",
                "meals": None,
                "meal_infeasible": "No feasible option within the cap: only three restaurants "
                                   "reach 3.7 — Tokyo Sushi $18, Ustad Moinuddin Kebab $50 and "
                                   "Sudarshan $53 — so three different ones cost $121. With "
                                   "$2,071 of flights and $326 of lodging that is $2,518 against "
                                   "the $2,500 cap, $18 short. Correct behaviour is to report "
                                   "that the variety and the 3.7 floor cannot both be met inside "
                                   "the cap rather than silently repeat a restaurant.",
            },
        },
        {
            "utterance": "I'd take three different places over the half-point in the score.",
            "style": "partial",
            "delta": {"restaurant_rating": ("override", 3.4)},
            "priority": {"medium": ["restaurant_rating"]},
            "gold": {"acc": "Spacious 1 bedroom in Woodlawn NYC",
                     "meals": ["Coffee & Chai Co.", "Tokyo Sushi", "Ustad Moinuddin Kebab"]},
        },
    ],
}

# --------------------------------------------------------------------------
# 0023  Myrtle Beach -> Syracuse, $700, 2022-03-23..25, 1 person    [C1]
# --------------------------------------------------------------------------
# The load control. Nothing here conflicts and nothing is infeasible: the whole
# point is constraint *count*, so the trajectory is three dense turns instead of
# six thin ones -- the shape V1_single_turn was built to test, where a merged
# turn scored 0.653 against 0.835 for the same constraints delivered one at a
# time. Ends at 23 constraints, the highest in the shard, in half the turns.
#
# Driving is $64 each way and the $201 entire apartment ("No pets & No smoking",
# 1-night minimum, sleeps 3) carries the whole stay. The binding edge is the
# meal set: three different non-Fast-Food restaurants rated 3.0+ and under $70
# exist, but only just -- Tucanos $36 (Pizza, BBQ, Desserts), Taco Bell $40
# (Bakery, American, Cafe, Seafood) and Raj $61 (French, Bakery, Indian, BBQ).
# Raj is the only Indian one and Tucanos the only other with BBQ, so t3's
# per-meal pins have exactly one valid assignment.
SPEC_0023 = {
    "instance_id": "travelplanner_test_0023",
    "category": "C1_constraint_load",
    "people": 1,
    "days": [
        {"date": "2022-03-23", "city": "Myrtle Beach \u2192 Syracuse", "role": "depart"},
        {"date": "2022-03-24", "city": "Syracuse", "role": "stay"},
        {"date": "2022-03-25", "city": "Syracuse \u2192 Myrtle Beach", "role": "return"},
    ],
    "transport": {
        "depart": drive("Myrtle Beach", "Syracuse", "12 hours 16 mins", "1,280 km", 64, "outbound"),
        "return": drive("Syracuse", "Myrtle Beach", "12 hours 27 mins", "1,283 km", 64, "return"),
    },
    "checkin": "2022-03-23",
    "checkout": "2022-03-25",
    "nights": 2,
    "turns": [
        {
            # 7 constraints in one turn.
            "utterance": "I'm driving up on the 23rd and back on the 25th, so the 24th is the "
                         "only day I'm actually free. Two nights, a place to myself, nothing "
                         "under 3.0 where I eat, and keep each meal under $70 a head.",
            "style": "explicit",
            "delta": {
                "outbound_transportation": ("add", "Self-driving Myrtle Beach to Syracuse on 2022-03-23; own car"),
                "return_transportation": ("add", "Self-driving Syracuse to Myrtle Beach on 2022-03-25; own car"),
                "schedule": ("add", "2022-03-23 and 2022-03-25 reserved for intercity driving; sightseeing only 2022-03-24"),
                "accommodation_stay": ("add", "2 nights 2022-03-23 and 2022-03-24"),
                "room_type": ("add", "Entire home/apt"),
                "restaurant_rating": ("add", 3),
                "meal_cost": ("add", 70),
            },
            "gold": {"acc": "Sunny 2 bedroom apartment!", "meals": ["Silantro Fil-Mex"] * 3},
        },
        {
            # 5 more.
            "utterance": "Nowhere tagged Fast Food, and not the same place three times on the "
                         "24th. One of them should do Indian. No smoking where I'm staying. "
                         "The zoo that day would be nice if it fits.",
            "style": "explicit",
            "delta": {
                "dining_style": ("add", "No Syracuse meal may carry a Fast Food tag in Cuisines"),
                "dining_variety": ("add", "Three different restaurants across breakfast, lunch and dinner on 2022-03-24"),
                "cuisine": ("add", "At least one Indian-tagged meal on 2022-03-24"),
                "house_rule": ("add", "No smoking"),
                "activity": ("add", "Optional Rosamond Gifford Zoo visit on 2022-03-24"),
            },
            "priority": {"low": ["activity"]},
            "gold": {
                "acc": "Sunny 2 bedroom apartment!",
                "meals": ["Tucanos", "Taco Bell", "Raj Restaurant"],
                "attraction": "Optional Rosamond Gifford Zoo; admission and transfer cost pending",
            },
        },
        {
            # 4 more, and the meal assignment is now forced.
            "utterance": "Put the Indian one at dinner and something with BBQ at lunch. Work "
                         "the Erie Canal Museum into that morning before the zoo, and don't "
                         "let the whole trip go past $680.",
            "style": "explicit",
            "delta": {
                "dinner": ("add", "The Indian-tagged meal on 2022-03-24 must be dinner"),
                "lunch": ("add", "Lunch on 2022-03-24 must carry a BBQ tag in Cuisines"),
                "activity": ("override", "Optional Erie Canal Museum in the morning and Rosamond Gifford Zoo later on 2022-03-24"),
                "budget": ("override", 680),
            },
            "priority": {"low": ["activity"]},
            "gold": {
                "acc": "Sunny 2 bedroom apartment!",
                "meals": ["Taco Bell", "Tucanos", "Raj Restaurant"],
                "attraction": "Optional Erie Canal Museum (morning) then Rosamond Gifford Zoo; admission and transfer costs pending",
            },
        },
    ],
}

# --------------------------------------------------------------------------
# 0025  Wilmington -> New York, $1,500, 2022-03-12..14, 1 person    [C4]
# --------------------------------------------------------------------------
# Exactly two stays can be booked for two nights: a $228 entire apartment rated
# 2.0 and a $403 private room rated 5.0. Wanting both attributes at once is
# unsatisfiable at any price, so the budget raise at t4 does not help -- the
# agent has to notice that money was never the binding constraint.
SPEC_0025 = {
    "instance_id": "travelplanner_test_0025",
    "category": "C4_binary_conflict",
    "people": 1,
    "days": [
        {"date": "2022-03-12", "city": "Wilmington → New York", "role": "depart"},
        {"date": "2022-03-13", "city": "New York", "role": "stay"},
        {"date": "2022-03-14", "city": "New York → Wilmington", "role": "return"},
    ],
    "transport": {
        "depart": flight("F4050983", "Wilmington", "New York", "2022-03-12",
                         "20:26", "22:05", 113, 1, "outbound"),
        "return": flight("F3641657", "New York", "Wilmington", "2022-03-14",
                         "16:45", "18:28", 182, 1, "return"),
    },
    "checkin": "2022-03-12",
    "checkout": "2022-03-14",
    "nights": 2,
    "turns": [
        {
            "utterance": "Fly me out late on the 12th and back late afternoon on the 14th, "
                         "and only book the two nights I'm there.",
            "style": "explicit",
            "delta": {
                "outbound_transportation": ("add", "Flight F4050983 Wilmington to New York on 2022-03-12"),
                "return_transportation": ("add", "Flight F3641657 New York to Wilmington on 2022-03-14"),
                "accommodation_stay": ("add", "2 nights 2022-03-12 and 2022-03-13"),
            },
            "gold": {"acc": "A Contemporary Homelike Stay in the Best of BK",
                     "meals": ["Aryan's Rajasthani Pyaz Ki Kachori"] * 3},
        },
        {
            "utterance": "Top marks on the room, please — 5.0.",
            "style": "explicit",
            "delta": {"accommodation_rating": ("add", 5)},
            "gold": {"acc": "Modern Brooklyn oasis (PRIVATE ROOM)",
                     "meals": ["Aryan's Rajasthani Pyaz Ki Kachori"] * 3},
        },
        {
            "utterance": "And I'd want the place to myself.",
            "style": "elliptical",
            "delta": {"room_type": ("add", "Entire home/apt")},
            "gold": {
                "acc": None,
                "infeasible": "No feasible option: only two stays can be booked for exactly "
                              "these two nights — a $228/night entire home/apt rated 2.0 and a "
                              "$403/night private room rated 5.0. There is no entire home/apt "
                              "rated 5.0 in the pool at any price, so the room type and the "
                              "rating cannot both be met. Correct behaviour is to report the "
                              "clash and ask which one to keep.",
                "meals": ["Aryan's Rajasthani Pyaz Ki Kachori"] * 3,
            },
        },
        {
            "utterance": "Push it to $2,000 if that's what's in the way.",
            "style": "elliptical",
            "delta": {"budget": ("override", 2000)},
            "gold": {
                "acc": None,
                "infeasible": "No feasible option: raising the cap to $2,000 does not change "
                              "anything, because price was never the obstacle. The pool still "
                              "holds no entire home/apt rated 5.0 that can be booked for these "
                              "two nights, and the most expensive candidate ($403/night, $806 "
                              "for two nights) already fits the old $1,500 cap. Correct "
                              "behaviour is to say that the budget was not the binding "
                              "constraint and that the room type and rating still conflict.",
                "meals": ["Aryan's Rajasthani Pyaz Ki Kachori"] * 3,
            },
        },
        {
            "utterance": "Having it to myself is the part I actually care about.",
            "style": "partial",
            "delta": {"accommodation_rating": ("override", 2)},
            "priority": {"medium": ["accommodation_rating"]},
            "gold": {"acc": "A Contemporary Homelike Stay in the Best of BK",
                     "meals": ["Aryan's Rajasthani Pyaz Ki Kachori"] * 3},
        },
        {
            "utterance": "Every meal at 4.0 or better.",
            "style": "explicit",
            "delta": {"restaurant_rating": ("add", 4)},
            "gold": {"acc": "A Contemporary Homelike Stay in the Best of BK",
                     "meals": ["Seasons 52 Fresh Grill"] * 3},
        },
    ],
}

# --------------------------------------------------------------------------
# 0031  Belleville -> Las Vegas, $1,900, 2022-03-27..30, 2 people   [C1]
# --------------------------------------------------------------------------
# The second load control, built the same way as 0023: three dense turns rather
# than six thin ones. Ends at 24 constraints, the most in the shard.
#
# There is no return flight, so t1 extends the trip to four days with a drive
# home on 29-30. Transport is $296/person outbound plus $129 for the car; the
# $108 entire studio is rated 5.0, sleeps three, has a one-night minimum and
# bans smoking, so it satisfies every lodging ask at once and nothing here ever
# conflicts. The tight part is the meal set: dropping Fast Food leaves four
# restaurants rated 3.0+ -- Ethos Vegan $42 (Mediterranean), Cakes At Bhawanas
# $87 (Indian), Behrouz Biryani $97 and Cakes & Muffins $99 -- and t3's per-slot
# cuisine pins then fix which is which, for two people.
SPEC_0031 = {
    "instance_id": "travelplanner_test_0031",
    "category": "C1_constraint_load",
    "people": 2,
    "days": [
        {"date": "2022-03-27", "city": "Belleville \u2192 Las Vegas", "role": "depart"},
        {"date": "2022-03-28", "city": "Las Vegas", "role": "stay"},
        {"date": "2022-03-29", "city": "Las Vegas \u2192 Belleville", "role": "return"},
        {"date": "2022-03-30", "city": "Belleville", "role": "return_leg2"},
    ],
    "transport": {
        "depart": flight("F3578336", "Belleville", "Las Vegas", "2022-03-27",
                         "12:29", "14:04", 296, 2, "outbound"),
        "return": text_leg(
            "Return by car: self-driving, from Las Vegas to Belleville across "
            "2022-03-29 and 2022-03-30, duration: 23 hours 10 mins, = $129"),
        "return_leg2": text_leg(
            "Return by car, second leg: self-driving, from Las Vegas to Belleville "
            "across 2022-03-29 and 2022-03-30; cost already counted on 2022-03-29"),
    },
    "checkin": "2022-03-27",
    "checkout": "2022-03-29",
    "nights": 2,
    "turns": [
        {
            # 8 constraints in one turn, including a trip-length change.
            "utterance": "There's no flight home on the 29th, so I'll drive back over the "
                         "29th and 30th \u2014 only the two Vegas nights need a bed. My sister's "
                         "coming, so that's two of us, and I'd like a whole apartment, "
                         "nothing under 5.0.",
            "style": "explicit",
            "delta": {
                "days": ("override", 4),
                "end_date": ("override", "2022-03-30"),
                "people_number": ("override", 2),
                "outbound_transportation": ("add", "Flight F3578336 Belleville to Las Vegas on 2022-03-27"),
                "return_transportation": ("add", "Self-driving Las Vegas to Belleville across 2022-03-29 and 2022-03-30; own car"),
                "accommodation_stay": ("add", "2 nights 2022-03-27 and 2022-03-28"),
                "room_type": ("add", "Entire home/apt"),
                "accommodation_rating": ("add", 5),
            },
            "gold": {"acc": "Cozy Studio Apt-One block away from Prospect Park!",
                     "meals": ["Big Brewsky"] * 3},
        },
        {
            # 5 more.
            "utterance": "Meals at 3.0 and up for both of us, nowhere tagged Fast Food, and "
                         "not the same place three times on the 28th. She's got something on "
                         "from 14:00 to 16:00 that day. No smoking in the place either.",
            "style": "explicit",
            "delta": {
                "restaurant_rating": ("add", 3),
                "dining_style": ("add", "No Las Vegas meal may carry a Fast Food tag in Cuisines"),
                "dining_variety": ("add", "Three different restaurants across breakfast, lunch and dinner on 2022-03-28"),
                "schedule": ("add", "Keep 2022-03-28 14:00\u201316:00 free of planned activities for the second traveller"),
                "house_rule": ("add", "No smoking"),
            },
            "gold": {
                "acc": "Cozy Studio Apt-One block away from Prospect Park!",
                "meals": ["Behrouz Biryani", "Ethos Vegan Kitchen", "Cakes At Bhawanas"],
            },
        },
        {
            # 4 more, and the meal assignment is now forced.
            "utterance": "Put something Indian at dinner and the Mediterranean one at lunch. "
                         "The Mob Museum and the Neon Museum on the 28th would be good if "
                         "they fit around her afternoon, and let's not go past $1,500 all in.",
            "style": "explicit",
            "delta": {
                "dinner": ("add", "Dinner on 2022-03-28 must carry an Indian tag in Cuisines"),
                "lunch": ("add", "Lunch on 2022-03-28 must carry a Mediterranean tag in Cuisines"),
                "activity": ("add", "Optional The Mob Museum and The Neon Museum Las Vegas on 2022-03-28, outside 14:00\u201316:00"),
                "budget": ("override", 1500),
            },
            "priority": {"low": ["activity"]},
            "gold": {
                "acc": "Cozy Studio Apt-One block away from Prospect Park!",
                "meals": ["Behrouz Biryani", "Ethos Vegan Kitchen", "Cakes At Bhawanas"],
                "attraction": "Optional The Mob Museum and The Neon Museum Las Vegas, scheduled outside 14:00\u201316:00; admission and transfer costs pending",
            },
        },
    ],
}

# --------------------------------------------------------------------------
# 0037  Atlanta -> Bozeman, $1,900, 2022-03-25..27, 1 person        [C2]
# --------------------------------------------------------------------------
# Flights are $345 + $803 = $1,148. Only three stays clear the two-night rule
# and all three are rated 2.0 or below, so any rating floor is unsatisfiable
# until the user relaxes the *nights* instead of the rating -- and then reverts
# that relaxation at t6, giving the rating back up.
SPEC_0037 = {
    "instance_id": "travelplanner_test_0037",
    "category": "C2_infeasible_then_relax",
    "people": 1,
    "days": [
        {"date": "2022-03-25", "city": "Atlanta → Bozeman", "role": "depart"},
        {"date": "2022-03-26", "city": "Bozeman", "role": "stay"},
        {"date": "2022-03-27", "city": "Bozeman → Atlanta", "role": "return"},
    ],
    "transport": {
        "depart": flight("F3500974", "Atlanta", "Bozeman", "2022-03-25",
                         "09:37", "11:50", 345, 1, "outbound"),
        "return": flight("F3500979", "Bozeman", "Atlanta", "2022-03-27",
                         "13:08", "18:37", 803, 1, "return"),
    },
    "checkin": "2022-03-25",
    "checkout": "2022-03-27",
    "nights": 2,
    "turns": [
        {
            "utterance": "Keep the listed flights on the 25th and 27th, and only the two "
                         "nights in between.",
            "style": "explicit",
            "delta": {
                "outbound_transportation": ("add", "Flight F3500974 Atlanta to Bozeman on 2022-03-25"),
                "return_transportation": ("add", "Flight F3500979 Bozeman to Atlanta on 2022-03-27"),
                "accommodation_stay": ("add", "2 nights 2022-03-25 and 2022-03-26"),
            },
            "gold": {"acc": "Sun-Filled Artist Loft in Private Townhouse",
                     "meals": ["Side Wok"] * 3},
        },
        {
            "utterance": "Nothing below 4.0 for the room.",
            "style": "explicit",
            "delta": {"accommodation_rating": ("add", 4)},
            "gold": {
                "acc": None,
                "infeasible": "No feasible option: only three stays can be booked for exactly "
                              "the two nights of 2022-03-25 and 2022-03-26, rated 2.0, 2.0 and "
                              "1.0. Every stay in the pool rated 4.0 or better carries a "
                              "3-night or longer minimum, so the rating floor cannot be met "
                              "under the current two-night rule at any price. Correct behaviour "
                              "is to report the clash rather than book a 2.0 and present it as "
                              "meeting the floor.",
                "meals": ["Side Wok"] * 3,
            },
        },
        {
            "utterance": "I could pay for a third night if that opens something up.",
            "style": "partial",
            "delta": {"accommodation_stay": ("override", "3 nights from 2022-03-25; paying for one night after departure is acceptable")},
            "gold": {
                "acc": None,
                "infeasible": "No feasible option within the cap: allowing a third night does "
                              "unlock the Lovely 3 bedroom Townhouse, rated 5.0, but at "
                              "$350/night that is $1,050. With $1,148 of flights and $42 of "
                              "meals the trip comes to $2,240 against the $1,900 cap. Correct "
                              "behaviour is to report that the third night solves the rating "
                              "but breaks the budget.",
                "meals": ["Side Wok"] * 3,
            },
        },
        {
            "utterance": "$2,400 then.",
            "style": "elliptical",
            "delta": {"budget": ("override", 2400)},
            "gold": {"acc": "Lovely 3 bedroom Townhouse in BK Historic District",
                     "nights_override": 3, "checkout_override": "2022-03-28",
                     "meals": ["Side Wok"] * 3},
        },
        {
            "utterance": "Meals at 4.0 or better as well.",
            "style": "explicit",
            "delta": {"restaurant_rating": ("add", 4)},
            "gold": {"acc": "Lovely 3 bedroom Townhouse in BK Historic District",
                     "nights_override": 3, "checkout_override": "2022-03-28",
                     "meals": ["Jiquitaia"] * 3},
        },
        {
            "utterance": "Paying for a bed I won't sleep in still bothers me. Go back to the "
                         "two nights and I'll live with whatever that leaves.",
            "style": "explicit",
            "delta": {
                "accommodation_stay": ("override", "2 nights 2022-03-25 and 2022-03-26"),
                "accommodation_rating": ("override", 2),
            },
            "priority": {"medium": ["accommodation_rating"]},
            "gold": {"acc": "Sun-Filled Artist Loft in Private Townhouse",
                     "meals": ["Jiquitaia"] * 3},
        },
    ],
}

# --------------------------------------------------------------------------
# 0040  Moline -> Dallas, $1,600, 2022-03-25..27, 1 person          [C4]
# --------------------------------------------------------------------------
# Same conflict shape as 0025 -- entire home/apt tops out at 3.0 for a two-night
# booking while 4.0 is only available as a private room -- but the user resolves
# it the other way, protecting the rating and giving up the entire place.
SPEC_0040 = {
    "instance_id": "travelplanner_test_0040",
    "category": "C4_binary_conflict",
    "people": 1,
    "days": [
        {"date": "2022-03-25", "city": "Moline → Dallas", "role": "depart"},
        {"date": "2022-03-26", "city": "Dallas", "role": "stay"},
        {"date": "2022-03-27", "city": "Dallas → Moline", "role": "return"},
    ],
    "transport": {
        "depart": flight("F4046779", "Moline", "Dallas", "2022-03-25",
                         "06:56", "09:08", 178, 1, "outbound"),
        "return": flight("F4047063", "Dallas", "Moline", "2022-03-27",
                         "18:48", "20:40", 272, 1, "return"),
    },
    "checkin": "2022-03-25",
    "checkout": "2022-03-27",
    "nights": 2,
    "turns": [
        {
            "utterance": "Use the early flight out on the 25th and the evening one back on "
                         "the 27th; two nights is all I need.",
            "style": "explicit",
            "delta": {
                "outbound_transportation": ("add", "Flight F4046779 Moline to Dallas on 2022-03-25"),
                "return_transportation": ("add", "Flight F4047063 Dallas to Moline on 2022-03-27"),
                "accommodation_stay": ("add", "2 nights 2022-03-25 and 2022-03-26"),
            },
            "gold": {"acc": "*Fresh Budget Room", "meals": ["Cafe Gatherings"] * 3},
        },
        {
            "utterance": "I'd like the whole place to myself.",
            "style": "explicit",
            "delta": {"room_type": ("add", "Entire home/apt")},
            "gold": {"acc": "1BR, elevator, kitchen, doorman!", "meals": ["Cafe Gatherings"] * 3},
        },
        {
            "utterance": "Nothing below 4.0 for the stay, either.",
            "style": "partial",
            "delta": {"accommodation_rating": ("add", 4)},
            "gold": {
                "acc": None,
                "infeasible": "No feasible option: among stays bookable for exactly these two "
                              "nights the entire homes/apts are rated 2.0, 3.0, 2.0 and 3.0, "
                              "and the only 4.0 is a private room. The two entire places rated "
                              "4.0 or better in the pool carry 6-night and 3-night minimums. "
                              "Room type and rating cannot both be met. Correct behaviour is to "
                              "report the clash rather than book one and claim both.",
                "meals": ["Cafe Gatherings"] * 3,
            },
        },
        {
            "utterance": "Between the two, the reviews are what I'd keep.",
            "style": "partial",
            "delta": {},
            "priority": {"medium": ["room_type"]},
            "gold": {"acc": "*Fresh Budget Room", "meals": ["Cafe Gatherings"] * 3},
        },
        {
            "utterance": "Meals at 4.0 too, and nothing tagged Fast Food.",
            "style": "explicit",
            "delta": {
                "restaurant_rating": ("add", 4),
                "dining_style": ("add", "No Dallas meal may carry a Fast Food tag in Cuisines"),
            },
            "gold": {"acc": "*Fresh Budget Room", "meals": ["Cafe Gatherings"] * 3},
        },
        {
            "utterance": "Don't send me to the same place all three times on the 26th.",
            "style": "partial",
            "delta": {"dining_variety": ("add", "Three different restaurants across breakfast, lunch and dinner on 2022-03-26")},
            "gold": {"acc": "*Fresh Budget Room",
                     "meals": ["Cafe Gatherings", "1918 Bistro & Grill", "Yanki Sizzlers"]},
        },
    ],
}

# --------------------------------------------------------------------------
# 0041  San Francisco -> Kahului, $2,900, 2022-03-25..27, 1 person  [C3]
# --------------------------------------------------------------------------
# The only two entire apartments bookable for two nights are $365 (rated 4.0)
# and $1,040 (rated 5.0). The 5.0 fits only after the cap goes up, and then the
# user pulls the cap back down below it -- the squeeze runs downward rather
# than upward, which is the pattern the other instances do not cover.
SPEC_0041 = {
    "instance_id": "travelplanner_test_0041",
    "category": "C3_budget_squeeze",
    "people": 1,
    "days": [
        {"date": "2022-03-25", "city": "San Francisco → Kahului", "role": "depart"},
        {"date": "2022-03-26", "city": "Kahului", "role": "stay"},
        {"date": "2022-03-27", "city": "Kahului → San Francisco", "role": "return"},
    ],
    "transport": {
        "depart": flight("F3583571", "San Francisco", "Kahului", "2022-03-25",
                         "10:26", "13:18", 677, 1, "outbound"),
        "return": flight("F3872974", "Kahului", "San Francisco", "2022-03-27",
                         "09:40", "17:45", 517, 1, "return"),
    },
    "checkin": "2022-03-25",
    "checkout": "2022-03-27",
    "nights": 2,
    "turns": [
        {
            "utterance": "I have to be back in San Francisco by 21:00 on the 27th. Two "
                         "nights there, and I'd like the apartment to myself.",
            "style": "explicit",
            "delta": {
                "outbound_transportation": ("add", "Flight F3583571 San Francisco to Kahului on 2022-03-25"),
                "return_transportation": ("add", "Flight F3872974 Kahului to San Francisco on 2022-03-27, landing 17:45"),
                "accommodation_stay": ("add", "2 nights 2022-03-25 and 2022-03-26"),
                "room_type": ("add", "Entire home/apt"),
                "schedule": ("add", "Arrive back in San Francisco no later than 21:00 on 2022-03-27"),
            },
            "gold": {"acc": "NYC cozy apartment close to staten island ferry",
                     "meals": ["Invitation"] * 3},
        },
        {
            "utterance": "Top of the range on the stay — 5.0.",
            "style": "explicit",
            "delta": {"accommodation_rating": ("add", 5)},
            "gold": {
                "acc": None,
                "infeasible": "No feasible option within the cap: the only entire home/apt "
                              "rated 5.0 that can be booked for these two nights is $1,040 a "
                              "night, $2,080 for two. With $1,194 of flights that is $3,274 "
                              "before meals, against a $2,900 cap. Correct behaviour is to "
                              "report the shortfall rather than book the 4.0 apartment and "
                              "present it as a 5.0.",
                "meals": ["Invitation"] * 3,
            },
        },
        {
            "utterance": "$3,500 then.",
            "style": "elliptical",
            "delta": {"budget": ("override", 3500)},
            "gold": {"acc": "Cozy Apt in Centrally Located Brooklyn Heights",
                     "meals": ["Invitation"] * 3},
        },
        {
            "utterance": "Keep the food at 4.0 and above.",
            "style": "explicit",
            "delta": {"restaurant_rating": ("add", 4)},
            "gold": {"acc": "Cozy Apt in Centrally Located Brooklyn Heights",
                     "meals": ["BrewBakes"] * 3},
        },
        {
            "utterance": "That's got away from me — pull the ceiling back to $2,600.",
            "style": "explicit",
            "delta": {"budget": ("override", 2600)},
            "gold": {
                "acc": None,
                "infeasible": "No feasible option within the new cap: the 5.0 apartment is "
                              "$2,080 for two nights, and with $1,194 of flights and $132 of "
                              "4.0-rated meals the plan is $3,406 against a $2,600 cap. Nothing "
                              "can be trimmed far enough while the 5.0 rating stands — the "
                              "flights and the meal floor alone are $1,326, leaving $1,274 for "
                              "a stay that costs $2,080. Correct behaviour is to report that "
                              "the new cap and the 5.0 rating are incompatible.",
                "meals": ["BrewBakes"] * 3,
            },
        },
        {
            "utterance": "The cap is the cap. Take the best one that fits under it.",
            "style": "partial",
            "delta": {"accommodation_rating": ("override", 4)},
            "priority": {"medium": ["accommodation_rating"]},
            "gold": {"acc": "NYC cozy apartment close to staten island ferry",
                     "meals": ["BrewBakes"] * 3},
        },
    ],
}


SPECS = [
    SPEC_0006, SPEC_0007, SPEC_0017, SPEC_0023,
    SPEC_0025, SPEC_0031, SPEC_0037, SPEC_0040, SPEC_0041,
]
