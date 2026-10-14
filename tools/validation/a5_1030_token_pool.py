#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Create a deterministic, varied token pool using the Attention tokenizer."""

import argparse
import hashlib
import json
from pathlib import Path

import httpx

MIN_POOL_TOKENS = 1024
TOKENIZE_TIMEOUT = 60
DIVERSE_PROMPT = """At dawn, a research vessel left the harbor to survey a coastal reef.
The crew checked weather forecasts, calibrated the depth sensor, and recorded the
water temperature at several stations. A marine biologist compared photographs
of coral colonies with observations from the previous season. Near the northern
channel, a school of small fish gathered above a patch of sea grass. The captain
slowed the vessel while the team lowered a sampling instrument. Each sample was
labeled with its location, collection time, and depth. Back at the laboratory,
the scientists would compare nutrient concentrations and estimate how currents
move sediment between the reef and the open sea.

In a mountain town, the public library organized an exhibition about local
history. Volunteers interviewed retired railway workers, scanned old maps, and
sorted photographs donated by families. One picture showed a wooden bridge that
had been replaced after a spring flood. Another captured the first electric tram
arriving at the market square. The exhibition paired these images with letters,
ticket stubs, and newspaper reports. Students prepared short descriptions for
visitors who were unfamiliar with the region. The librarian asked them to
distinguish documented facts from recollections. Several residents offered
corrections, helping the team produce a more accurate account of everyday life.

A software team investigated a delayed delivery notification. They traced one
order from the checkout page through the payment service and the warehouse
queue. The payment succeeded, but a retry sent the same message twice. One
consumer recognized the duplicate while another created an extra delivery task.
The engineers added an order identifier to the records and reviewed how long
those identifiers remained available. They then tested delayed messages,
restarts, and temporary network failures. Their report explained which events
could safely be repeated and which required reconciliation. After the change,
the dashboard showed completed orders, pending work, and duplicate messages as
separate counts.

An urban planner walked along a busy avenue with residents from the neighborhood.
They measured crossing distances, noted the location of bus stops, and counted
cyclists during the morning commute. Parents described a corner where parked
vehicles blocked the view of approaching traffic. Shop owners wanted delivery
spaces close to their entrances. The planner sketched several options, including
wider sidewalks, a protected cycle lane, and a shorter pedestrian crossing. Each
option required a different arrangement of parking and loading areas. Before
choosing a design, the city would collect observations at different times of day
and invite residents to discuss the tradeoffs in a public meeting.

Inside a bakery, the morning shift prepared three kinds of bread. The baker
weighed flour, water, salt, and yeast before adjusting the mixing time for the
room temperature. A dough made with whole grains absorbed more water than the
white flour batch. After resting, each portion was shaped and placed in a covered
tray. The apprentice checked the rise by gently pressing the surface rather than
relying only on the clock. Steam entered the oven during the first minutes of
baking. When the loaves cooled, the team compared crust color, crumb structure,
and flavor, recording which changes produced a consistent result.

An astronomy club met outside the city to observe the night sky. Members chose a
site with a clear southern horizon and little artificial light. Before sunset,
they aligned their telescopes and explained how to read a star chart. Clouds
briefly covered the western sky, so the group changed the observation order.
Through a small telescope, a planet appeared as a bright disk with several faint
points nearby. A larger instrument revealed a cluster of stars that looked like
a hazy patch to the unaided eye. Observers recorded the time, equipment, and sky
conditions so that they could compare their notes at the next meeting.

A gardener planned a vegetable plot beside a community center. The soil near the
fence was compacted, while the beds farther south drained quickly after rain.
She collected samples from both areas and discussed the results with other
volunteers. They added compost, repaired a leaking water pipe, and arranged paths
wide enough for a wheelbarrow. Leafy vegetables were planted where afternoon
shade kept the ground cooler. Beans climbed a simple wooden frame, leaving more
space for smaller plants below. The group kept a notebook of planting dates,
rainfall, pests, and harvests. Those records would guide the following season
without assuming that every year has the same weather.

A museum conservator examined a painted wooden chest before placing it on
display. The surface had small cracks, loose pigment, and traces of an older
repair. She photographed the damage under different lighting and compared the
materials with other objects from the same period. A curator checked the
ownership records while a carpenter inspected the joints. The team chose a
support that would distribute the weight evenly and reduce stress on the lid.
They also specified a stable humidity range for the gallery. A label described
the chest's construction and use, while the treatment record documented which
parts had been stabilized and which remained untouched.

At a regional station, a dispatcher reviewed the timetable for the evening
trains. A freight service had arrived late, and maintenance had closed one track
near the eastern junction. The dispatcher considered where passenger trains
could pass each other without blocking the platform. Station staff informed
travelers about the revised departure times and arranged assistance for people
with connecting services. Once the freight train cleared the junction, the team
updated the route assignments. They compared planned and actual arrival times
after the shift, distinguishing delays caused by the closure from those caused
by boarding. The review helped them prepare a more practical contingency plan.

A materials researcher compared two methods of making a lightweight panel. One
method used a woven reinforcement, while the other arranged short fibers in
several directions. She prepared samples of equal thickness and measured their
mass before testing them under bending loads. Some samples failed suddenly;
others developed visible cracks before losing strength. The researcher repeated
the measurements with a second batch and recorded variation between specimens.
She also checked how moisture affected the results. The final comparison
included manufacturing time, material consumption, stiffness, and failure
behavior. It described the limits of the experiment so that designers could
decide whether either method suited a particular application.

A music teacher helped a small ensemble prepare for a concert. The violinists
practiced a passage slowly while the cellist marked the changes in rhythm. A
flutist suggested a quieter opening so that the melody would remain clear. The
teacher asked the players to listen for the end of each phrase rather than
watching only their own notes. During rehearsal, they tried different seating
arrangements and compared the balance of the instruments. They recorded one
performance, then discussed timing and articulation together. At the next
session, the ensemble focused on the transitions between sections before playing
the complete piece without interruption.

An archivist received several boxes of correspondence from a former engineering
office. The papers included sketches, meeting notes, photographs, and invoices.
Some pages had dates but no names; others referred to projects whose titles had
changed over time. She grouped the documents by source and preserved their
original order where possible. A searchable catalog recorded descriptions,
dates, and relationships between folders. Fragile sheets were stored separately
with references connecting them to the main collection. Researchers could use
the catalog to locate relevant material without repeatedly handling every page.
The archivist documented uncertain identifications so that later discoveries
could improve the records without obscuring the earlier evidence.
"""


def create_token_pool(args: argparse.Namespace) -> dict:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    request = {
        "model": args.model,
        "prompt": DIVERSE_PROMPT,
        "add_special_tokens": False,
    }
    (output / "tokenize-request.json").write_text(
        json.dumps(request, ensure_ascii=False, indent=2) + "\n"
    )
    with httpx.Client(timeout=TOKENIZE_TIMEOUT, trust_env=False) as client:
        response = client.post(
            f"{args.attention_url.rstrip('/')}/tokenize", json=request
        )
    (output / "tokenize.response.body").write_bytes(response.content)
    response.raise_for_status()
    data = response.json()
    tokens = data["tokens"]
    if (
        not isinstance(tokens, list)
        or len(tokens) < MIN_POOL_TOKENS
        or any(type(token) is not int or token < 0 for token in tokens)
    ):
        raise ValueError("tokenizer must return at least 1024 nonnegative integer IDs")
    pool_text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    (output / "token-pool.json").write_text(pool_text)
    metadata = {
        "input_kind": "diverse-prose-v1",
        "model": args.model,
        "attention_url": args.attention_url,
        "token_count": len(tokens),
        "long_input_tokens": MIN_POOL_TOKENS,
        "prompt_sha256": hashlib.sha256(DIVERSE_PROMPT.encode()).hexdigest(),
        "pool_sha256": hashlib.sha256(pool_text.encode()).hexdigest(),
        "long_input_unique_token_count": len(set(tokens[:MIN_POOL_TOKENS])),
    }
    (output / "token-pool.metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attention-url", required=True)
    parser.add_argument(
        "--output", required=True, help="new directory for pool artifacts"
    )
    parser.add_argument("--model", default="dsv4-afd")
    args = parser.parse_args()
    metadata = create_token_pool(args)
    print(
        f"Created varied pool ({metadata['token_count']} tokens); "
        f"use --token-pool {Path(args.output) / 'token-pool.json'}"
    )


if __name__ == "__main__":
    main()
