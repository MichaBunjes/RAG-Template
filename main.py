import asyncio
import logging
import os

from flask import Flask, jsonify, request
from flask_cors import CORS
from marshmallow import Schema, ValidationError, fields

from query_handler import QueryHandler


class RagSystem:
    async def __init__(self) -> None:
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.QueryHandler = await QueryHandler.create()


class ChatInputSchema(Schema):
    user_question = fields.Str(required=True)
    messages = fields.List(fields.Dict(), required=True)


class FeedbackInputSchema(Schema):
    is_positive = fields.Bool(required=True)
    feedback_text = fields.Str(required=True)
    messages = fields.List(fields.Dict(), required=True)


app = Flask(__name__)

# Configure CORS for local development
origins = [
    "http://localhost:5173",  # Vite default port
    "http://127.0.0.1:5173",
]

CORS(
    app,
    resources={
        r"/*": {
            "origins": origins,
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type"],
        }
    },
)

# Setup RAG
global RagBackend
RagBackend = RagSystem()


@app.route("/", methods=["POST"])
def process_chat_json():
    try:
        chat_input = ChatInputSchema().load(request.get_json())

        user_question = chat_input["user_question"]
        messages = chat_input["messages"]

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        response, _ = loop.run_until_complete(
            RagBackend.QueryHandler.handle_query(user_question, messages)
        )

        return jsonify(response)
    except ValidationError as err:
        return jsonify(err.messages), 400


@app.route("/feedback", methods=["POST"])
def process_feedback_json():
    try:
        feedback_input = FeedbackInputSchema().load(request.get_json())

        feedback = {
            "is_positive": feedback_input["is_postitive"],
            "feedback_text": feedback_input["feedback_text"],
            "messages": feedback_input["messages"],
        }

        # FeedbackHandler.save_feedback(feedback)
        return jsonify({"message": "Feedback submitted successfully."})
    except ValidationError as err:
        return jsonify(err.messages), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
