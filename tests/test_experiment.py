
import json
from datetime import datetime, timezone

from samaritan.contracts import Source, new_id
from samaritan.database import Database
from samaritan.ingestion import IngestionService
from samaritan.repository import Repository


def test_synthetic_cyber_telemetry_end_to_end(tmp_path):
    database = Database(str(tmp_path / "experiment.db"))
    database.initialize()

    repository = Repository(database)

    source = Source(
        source_id=new_id(),
        source_type="synthetic",
        name="M1.6 Synthetic Telemetry",
        created_at=datetime(2026, 11, 16, 8, 0, tzinfo=timezone.utc),
    )
    repository.register_source(source)

    service = IngestionService(repository)

    ip = ("ip_address", "192.0.2.10")
    user = ("user_account", "synthetic-user")
    host = ("hostname", "workstation-01.example.test")

    def submit(received_at, event_specs):
        entity_ids = {}

        for spec in event_specs:
            for entity_type, identifier, _ in spec["participants"]:
                key = (entity_type, identifier)
                if key not in entity_ids:
                    entity_ids[key] = new_id()

        entities = [
            {
                "entity_id": entity_id,
                "entity_type": key[0],
                "canonical_identifier": key[1],
            }
            for key, entity_id in entity_ids.items()
        ]

        events = []

        for spec in event_specs:
            participants = [
                {
                    "entity_id": entity_ids[(entity_type, identifier)],
                    "role": role,
                }
                for entity_type, identifier, role
                in spec["participants"]
            ]

            events.append({
                "event_id": new_id(),
                "event_type": spec["event_type"],
                "original_timestamp": spec["timestamp"],
                "normalized_data": spec["data"],
                "participants": participants,
            })

        envelope = {
            "format_version": 1,
            "entities": entities,
            "events": events,
            "relationships": [],
        }

        raw_payload = json.dumps(
            envelope,
            sort_keys=True,
            separators=(",", ":"),
        )

        result = service.ingest(
            source.source_id,
            raw_payload,
            received_at,
        )

        assert result.accepted is True
        return raw_payload, result

    first_raw, first_result = submit(
        datetime(2026, 11, 16, 9, 3, tzinfo=timezone.utc),
        [
            {
                "event_type": "authentication_failure",
                "timestamp": f"2026-11-16T09:0{minute}:00Z",
                "data": {
                    "username": "synthetic-user",
                    "result": "failure",
                },
                "participants": [
                    (*ip, "source"),
                    (*user, "target"),
                ],
            }
            for minute in (0, 1, 2)
        ],
    )

    second_raw, second_result = submit(
        datetime(2026, 11, 16, 9, 6, tzinfo=timezone.utc),
        [
            {
                "event_type": "authentication_success",
                "timestamp": "2026-11-16T09:05:00Z",
                "data": {
                    "username": "synthetic-user",
                    "result": "success",
                },
                "participants": [
                    (*ip, "source"),
                    (*user, "target"),
                ],
            }
        ],
    )

    third_raw, third_result = submit(
        datetime(2026, 11, 16, 9, 9, tzinfo=timezone.utc),
        [
            {
                "event_type": "host_activity",
                "timestamp": "2026-11-16T09:08:00Z",
                "data": {
                    "activity": "interactive_session_started",
                },
                "participants": [
                    (*user, "subject"),
                    (*host, "target"),
                ],
            }
        ],
    )

    results = [first_result, second_result, third_result]
    raw_payloads = [first_raw, second_raw, third_raw]

    assert len({
        result.observation_id for result in results
    }) == 3

    observations = repository.list_observations(
        source_id=source.source_id,
    )

    assert len(observations) == 3
    assert [item.raw_payload for item in observations] == raw_payloads

    retrieved_events = repository.list_events()

    assert len(retrieved_events) == 5

    assert [
        event.event_type for event in retrieved_events
    ] == [
        "authentication_failure",
        "authentication_failure",
        "authentication_failure",
        "authentication_success",
        "host_activity",
    ]

    assert [
        event.occurred_at.minute for event in retrieved_events
    ] == [0, 1, 2, 5, 8]

    expected_observation_ids = [
        first_result.observation_id,
        first_result.observation_id,
        first_result.observation_id,
        second_result.observation_id,
        third_result.observation_id,
    ]

    assert [
        event.observation_id for event in retrieved_events
    ] == expected_observation_ids

    stored_ip = repository.get_entity_by_identity(*ip)
    stored_user = repository.get_entity_by_identity(*user)
    stored_host = repository.get_entity_by_identity(*host)

    assert stored_ip is not None
    assert stored_user is not None
    assert stored_host is not None

    for event in retrieved_events[:4]:
        participant_ids = {
            participant.entity_id
            for participant in event.participants
        }
        assert stored_ip.entity_id in participant_ids
        assert stored_user.entity_id in participant_ids

    final_participant_ids = {
        participant.entity_id
        for participant in retrieved_events[4].participants
    }

    assert stored_user.entity_id in final_participant_ids
    assert stored_host.entity_id in final_participant_ids

    with database.connection() as connection:
        entity_count = connection.execute(
            "SELECT COUNT(*) FROM entities"
        ).fetchone()[0]

        observation_count = connection.execute(
            "SELECT COUNT(*) FROM observations"
        ).fetchone()[0]

        event_count = connection.execute(
            "SELECT COUNT(*) FROM events"
        ).fetchone()[0]

    assert entity_count == 3
    assert observation_count == 3
    assert event_count == 5

