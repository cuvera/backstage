from app.messaging.producers.painpoint_producer import send_painpoint_captured


if __name__ == "__main__":
    msg_id = send_painpoint_captured(
        tenant_id="acme",
        user_id="u_126t3",
        raw_text="manager is scolding very hard everydaymangagers on daily basis bcs of miss communications" \
        "",
        session_id="s_teeeest",
        department="Manufacturing",
        source="chat",
        metadata={"locale": "en-IN", "env": "dev"},
    )
    print("Sent message_id:", msg_id)
