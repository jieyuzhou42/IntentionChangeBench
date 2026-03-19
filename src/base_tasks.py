from __future__ import annotations

from typing import Dict, List

from models import BaseTask


def get_scheduling_tasks() -> List[BaseTask]:
    return [
        BaseTask(
            instance_id="scheduling_001",
            task_type="scheduling",
            subtype="team_meeting",
            world_state={
                "available_slots": ["Mon 09:00", "Mon 14:00", "Tue 11:00", "Wed 15:00"],
                "locations": ["Room A", "Room B", "Zoom"],
            },
            initial_intention={
                "goal": "schedule_team_meeting",
                "day_preference": "Mon",
                "time_preference": "afternoon",
                "avoid_days": ["Wed"],
                "location": "Zoom",
                "priority": ["avoid_days", "time_preference", "location"],
            },
        ),
        BaseTask(
            instance_id="scheduling_002",
            task_type="scheduling",
            subtype="advisor_meeting",
            world_state={
                "available_slots": ["Tue 10:00", "Tue 16:00", "Thu 13:00", "Fri 09:00"],
                "locations": ["Office", "Zoom"],
            },
            initial_intention={
                "goal": "schedule_advisor_meeting",
                "day_preference": "Thu",
                "time_preference": "afternoon",
                "avoid_days": ["Fri"],
                "location": "Office",
                "priority": ["location", "day_preference", "time_preference"],
            },
        ),
        BaseTask(
            instance_id="scheduling_003",
            task_type="scheduling",
            subtype="doctor_visit",
            world_state={
                "available_slots": ["Mon 08:30", "Tue 09:30", "Thu 15:30", "Fri 14:00"],
                "locations": ["Clinic East", "Clinic West"],
            },
            initial_intention={
                "goal": "schedule_doctor_visit",
                "day_preference": "Fri",
                "time_preference": "afternoon",
                "avoid_days": ["Mon"],
                "location": "Clinic East",
                "priority": ["day_preference", "location", "time_preference"],
            },
        ),
        BaseTask(
            instance_id="scheduling_004",
            task_type="scheduling",
            subtype="interview_slot",
            world_state={
                "available_slots": ["Wed 10:00", "Wed 14:30", "Thu 11:00", "Fri 16:00"],
                "locations": ["Virtual", "HQ"],
            },
            initial_intention={
                "goal": "schedule_interview",
                "day_preference": "Wed",
                "time_preference": "morning",
                "avoid_days": ["Fri"],
                "location": "Virtual",
                "priority": ["time_preference", "location", "avoid_days"],
            },
        ),
        BaseTask(
            instance_id="scheduling_005",
            task_type="scheduling",
            subtype="study_group",
            world_state={
                "available_slots": ["Tue 18:00", "Wed 18:30", "Thu 17:00", "Sun 13:00"],
                "locations": ["Library", "Cafe", "Zoom"],
            },
            initial_intention={
                "goal": "schedule_study_group",
                "day_preference": "Thu",
                "time_preference": "evening",
                "avoid_days": ["Sun"],
                "location": "Library",
                "priority": ["location", "day_preference", "time_preference"],
            },
        ),
        BaseTask(
            instance_id="scheduling_006",
            task_type="scheduling",
            subtype="project_sync",
            world_state={
                "available_slots": ["Mon 12:00", "Tue 14:00", "Thu 12:30", "Fri 15:00"],
                "locations": ["Room C", "Zoom"],
            },
            initial_intention={
                "goal": "schedule_project_sync",
                "day_preference": "Tue",
                "time_preference": "afternoon",
                "avoid_days": ["Mon"],
                "location": "Room C",
                "priority": ["day_preference", "time_preference", "location"],
            },
        ),
        BaseTask(
            instance_id="scheduling_007",
            task_type="scheduling",
            subtype="lunch_meeting",
            world_state={
                "available_slots": ["Mon 12:00", "Tue 12:30", "Wed 13:00", "Thu 11:30"],
                "locations": ["Cafe 1", "Cafe 2"],
            },
            initial_intention={
                "goal": "schedule_lunch_meeting",
                "day_preference": "Tue",
                "time_preference": "midday",
                "avoid_days": ["Wed"],
                "location": "Cafe 1",
                "priority": ["location", "avoid_days", "day_preference"],
            },
        ),
        BaseTask(
            instance_id="scheduling_008",
            task_type="scheduling",
            subtype="campus_tour",
            world_state={
                "available_slots": ["Fri 10:00", "Fri 14:00", "Sat 09:00", "Sat 13:30"],
                "locations": ["North Gate", "Main Lobby"],
            },
            initial_intention={
                "goal": "schedule_campus_tour",
                "day_preference": "Sat",
                "time_preference": "morning",
                "avoid_days": ["Fri"],
                "location": "North Gate",
                "priority": ["day_preference", "time_preference", "location"],
            },
        ),
        BaseTask(
            instance_id="scheduling_009",
            task_type="scheduling",
            subtype="fitness_class",
            world_state={
                "available_slots": ["Mon 07:00", "Tue 18:00", "Thu 19:00", "Sat 10:00"],
                "locations": ["Gym A", "Gym B"],
            },
            initial_intention={
                "goal": "schedule_fitness_class",
                "day_preference": "Thu",
                "time_preference": "evening",
                "avoid_days": ["Mon"],
                "location": "Gym A",
                "priority": ["time_preference", "location", "day_preference"],
            },
        ),
        BaseTask(
            instance_id="scheduling_010",
            task_type="scheduling",
            subtype="parent_call",
            world_state={
                "available_slots": ["Sun 09:00", "Sun 20:00", "Sat 21:00", "Fri 19:30"],
                "locations": ["Phone", "Zoom"],
            },
            initial_intention={
                "goal": "schedule_parent_call",
                "day_preference": "Sun",
                "time_preference": "evening",
                "avoid_days": ["Fri"],
                "location": "Phone",
                "priority": ["day_preference", "time_preference", "location"],
            },
        ),
    ]


def get_retrieval_tasks() -> List[BaseTask]:
    return [
        BaseTask(
            instance_id="retrieval_001",
            task_type="retrieval_ranking",
            subtype="hotel_search",
            world_state={
                "candidate_items": [
                    {"id": "h1", "area": "downtown", "price": 220, "rating": 4.2, "pet_friendly": False},
                    {"id": "h2", "area": "midtown", "price": 245, "rating": 4.7, "pet_friendly": True},
                ]
            },
            initial_intention={
                "goal": "find_hotel",
                "city": "Boston",
                "budget_max": 250,
                "area": "downtown",
                "pet_friendly": False,
                "priority": ["area", "budget_max", "rating"],
            },
        ),
        BaseTask(
            instance_id="retrieval_002",
            task_type="retrieval_ranking",
            subtype="flight_search",
            world_state={
                "candidate_items": [
                    {"id": "f1", "stops": 0, "price": 380, "airline": "Delta", "arrival_time": "morning"},
                    {"id": "f2", "stops": 1, "price": 290, "airline": "United", "arrival_time": "evening"},
                ]
            },
            initial_intention={
                "goal": "find_flight",
                "origin": "ATL",
                "destination": "SFO",
                "budget_max": 400,
                "nonstop": True,
                "arrival_time": "morning",
                "priority": ["nonstop", "arrival_time", "budget_max"],
            },
        ),
        BaseTask(
            instance_id="retrieval_003",
            task_type="retrieval_ranking",
            subtype="restaurant_search",
            world_state={
                "candidate_items": [
                    {"id": "r1", "cuisine": "Italian", "distance": 1.2, "price_level": 2, "rating": 4.4},
                    {"id": "r2", "cuisine": "Japanese", "distance": 3.5, "price_level": 3, "rating": 4.8},
                ]
            },
            initial_intention={
                "goal": "find_restaurant",
                "cuisine": "Italian",
                "distance_max": 2.0,
                "price_level_max": 2,
                "priority": ["cuisine", "distance_max", "rating"],
            },
        ),
        BaseTask(
            instance_id="retrieval_004",
            task_type="retrieval_ranking",
            subtype="paper_search",
            world_state={
                "candidate_items": [
                    {"id": "p1", "venue": "CHI", "year": 2024, "topic": "AI agents", "citations": 12},
                    {"id": "p2", "venue": "UIST", "year": 2025, "topic": "interactive systems", "citations": 4},
                ]
            },
            initial_intention={
                "goal": "find_paper",
                "topic": "AI agents",
                "venue": "CHI",
                "year_min": 2023,
                "priority": ["topic", "venue", "year_min"],
            },
        ),
        BaseTask(
            instance_id="retrieval_005",
            task_type="retrieval_ranking",
            subtype="job_search",
            world_state={
                "candidate_items": [
                    {"id": "j1", "location": "New York", "salary": 150000, "remote": False, "level": "intern"},
                    {"id": "j2", "location": "Remote", "salary": 135000, "remote": True, "level": "intern"},
                ]
            },
            initial_intention={
                "goal": "find_job",
                "role": "ML Engineer Intern",
                "location": "New York",
                "remote": False,
                "priority": ["role", "location", "salary"],
            },
        ),
        BaseTask(
            instance_id="retrieval_006",
            task_type="retrieval_ranking",
            subtype="apartment_search",
            world_state={
                "candidate_items": [
                    {"id": "a1", "area": "Midtown", "rent": 1800, "pets": True, "laundry": True},
                    {"id": "a2", "area": "Downtown", "rent": 1600, "pets": False, "laundry": True},
                ]
            },
            initial_intention={
                "goal": "find_apartment",
                "area": "Midtown",
                "rent_max": 1900,
                "pets": True,
                "priority": ["area", "pets", "rent_max"],
            },
        ),
        BaseTask(
            instance_id="retrieval_007",
            task_type="retrieval_ranking",
            subtype="laptop_search",
            world_state={
                "candidate_items": [
                    {"id": "l1", "brand": "Lenovo", "weight_lb": 3.1, "price": 1100, "ram_gb": 16},
                    {"id": "l2", "brand": "Dell", "weight_lb": 4.3, "price": 900, "ram_gb": 16},
                ]
            },
            initial_intention={
                "goal": "find_laptop",
                "ram_gb_min": 16,
                "weight_lb_max": 3.5,
                "budget_max": 1200,
                "priority": ["weight_lb_max", "ram_gb_min", "budget_max"],
            },
        ),
        BaseTask(
            instance_id="retrieval_008",
            task_type="retrieval_ranking",
            subtype="scholarship_search",
            world_state={
                "candidate_items": [
                    {"id": "s1", "field": "CS", "amount": 5000, "international_ok": True},
                    {"id": "s2", "field": "Design", "amount": 7000, "international_ok": False},
                ]
            },
            initial_intention={
                "goal": "find_scholarship",
                "field": "CS",
                "international_ok": True,
                "priority": ["international_ok", "amount", "field"],
            },
        ),
        BaseTask(
            instance_id="retrieval_009",
            task_type="retrieval_ranking",
            subtype="conference_search",
            world_state={
                "candidate_items": [
                    {"id": "c1", "field": "HCI", "location": "US", "deadline_month": "Apr"},
                    {"id": "c2", "field": "AI", "location": "Europe", "deadline_month": "May"},
                ]
            },
            initial_intention={
                "goal": "find_conference",
                "field": "HCI",
                "location": "US",
                "priority": ["field", "deadline_month", "location"],
            },
        ),
        BaseTask(
            instance_id="retrieval_010",
            task_type="retrieval_ranking",
            subtype="course_search",
            world_state={
                "candidate_items": [
                    {"id": "c101", "topic": "NLP", "time": "morning", "mode": "in-person"},
                    {"id": "c102", "topic": "ML systems", "time": "afternoon", "mode": "online"},
                ]
            },
            initial_intention={
                "goal": "find_course",
                "topic": "NLP",
                "time": "afternoon",
                "mode": "in-person",
                "priority": ["topic", "time", "mode"],
            },
        ),
    ]


def get_transaction_tasks() -> List[BaseTask]:
    return [
        BaseTask(
            instance_id="transaction_001",
            task_type="transaction",
            subtype="book_flight",
            world_state={"payment_methods": ["Visa", "Amex"], "seat_options": ["aisle", "window"]},
            initial_intention={
                "goal": "book_flight",
                "destination": "San Francisco",
                "date": "2026-05-12",
                "seat": "window",
                "payment_method": "Visa",
                "priority": ["date", "seat", "payment_method"],
            },
        ),
        BaseTask(
            instance_id="transaction_002",
            task_type="transaction",
            subtype="book_hotel",
            world_state={"room_types": ["queen", "king"], "payment_methods": ["Visa", "Mastercard"]},
            initial_intention={
                "goal": "book_hotel",
                "city": "Chicago",
                "check_in": "2026-06-10",
                "room_type": "king",
                "payment_method": "Mastercard",
                "priority": ["check_in", "room_type", "payment_method"],
            },
        ),
        BaseTask(
            instance_id="transaction_003",
            task_type="transaction",
            subtype="order_laptop",
            world_state={"colors": ["silver", "black"], "shipping": ["standard", "express"]},
            initial_intention={
                "goal": "order_laptop",
                "model": "ThinkPad X1",
                "color": "black",
                "shipping": "standard",
                "priority": ["model", "shipping", "color"],
            },
        ),
        BaseTask(
            instance_id="transaction_004",
            task_type="transaction",
            subtype="register_workshop",
            world_state={"ticket_types": ["student", "general"], "delivery": ["email", "sms"]},
            initial_intention={
                "goal": "register_workshop",
                "workshop": "Agent Evaluation",
                "ticket_type": "student",
                "delivery": "email",
                "priority": ["ticket_type", "delivery"],
            },
        ),
        BaseTask(
            instance_id="transaction_005",
            task_type="transaction",
            subtype="order_food",
            world_state={"sizes": ["small", "medium", "large"], "delivery_modes": ["pickup", "delivery"]},
            initial_intention={
                "goal": "order_food",
                "item": "pepperoni pizza",
                "size": "medium",
                "delivery_mode": "delivery",
                "priority": ["delivery_mode", "size"],
            },
        ),
        BaseTask(
            instance_id="transaction_006",
            task_type="transaction",
            subtype="book_train",
            world_state={"seat_options": ["window", "aisle"], "classes": ["economy", "business"]},
            initial_intention={
                "goal": "book_train",
                "route": "Boston to NYC",
                "date": "2026-04-18",
                "class": "economy",
                "seat": "aisle",
                "priority": ["date", "class", "seat"],
            },
        ),
        BaseTask(
            instance_id="transaction_007",
            task_type="transaction",
            subtype="submit_application",
            world_state={"document_options": ["resume_v1", "resume_v2"], "contact_methods": ["email", "phone"]},
            initial_intention={
                "goal": "submit_application",
                "position": "Research Intern",
                "resume": "resume_v1",
                "contact_method": "email",
                "priority": ["position", "resume", "contact_method"],
            },
        ),
        BaseTask(
            instance_id="transaction_008",
            task_type="transaction",
            subtype="book_saloon",
            world_state={"service_types": ["haircut", "color"], "payment_methods": ["cash", "card"]},
            initial_intention={
                "goal": "book_salon",
                "service": "haircut",
                "date": "2026-04-02",
                "payment_method": "card",
                "priority": ["date", "service", "payment_method"],
            },
        ),
        BaseTask(
            instance_id="transaction_009",
            task_type="transaction",
            subtype="reserve_rental_car",
            world_state={"car_types": ["sedan", "SUV"], "insurance_options": ["basic", "full"]},
            initial_intention={
                "goal": "reserve_rental_car",
                "pickup_date": "2026-07-01",
                "car_type": "sedan",
                "insurance": "basic",
                "priority": ["pickup_date", "car_type", "insurance"],
            },
        ),
        BaseTask(
            instance_id="transaction_010",
            task_type="transaction",
            subtype="purchase_ticket",
            world_state={"sections": ["front", "middle", "back"], "delivery": ["mobile", "print"]},
            initial_intention={
                "goal": "purchase_ticket",
                "event": "concert",
                "section": "middle",
                "delivery": "mobile",
                "priority": ["section", "delivery"],
            },
        ),
    ]


def get_all_base_tasks() -> Dict[str, List[BaseTask]]:
    return {
        "scheduling": get_scheduling_tasks(),
        "retrieval_ranking": get_retrieval_tasks(),
        "transaction": get_transaction_tasks(),
    }
