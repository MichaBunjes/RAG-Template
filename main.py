import asyncio
import logging
import os

from flask import Flask, jsonify, request
from marshmallow import Schema, ValidationError, fields
from query_handler import QueryHandler


class RagSystem:
    def __init__(self) -> None:
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.QueryHandler = asyncio.run(QueryHandler.create())


class UserInputSchema(Schema):
    request_type = fields.Str(required=True)
    user_question = fields.Str(required=True)
    messages = fields.List(fields.Dict(), required=True)


class FeedbackInputSchema(Schema):
    request_type = fields.Str(required=True)
    is_positive = fields.Bool(required=True)
    feedback_text = fields.Str(required=True)
    messages = fields.List(fields.Dict(), required=True)


app = Flask(__name__)

# Setup RAG
global RagBackend
RagBackend = RagSystem()


@app.route("/", methods=["POST", "OPTIONS"])
def process_chat_json():
    if request.method == "OPTIONS":
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Max-Age": "3600",
        }

        return jsonify(""), 204, headers

    if request.method == "POST":
        # Set CORS headers for the main request
        headers = {"Access-Control-Allow-Origin": "*"}
        try:
            request_data = request.get_json()

            if "request_type" not in request_data:
                return jsonify({"error": "Missing request_type field"}), 400, headers
            request_type = request_data["request_type"]

            if request_type == "rag_query":
                user_input = UserInputSchema().load(request_data)

                user_question = user_input["user_question"]
                messages = user_input["messages"]

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                response = loop.run_until_complete(
                    RagBackend.QueryHandler.handle_query(user_question, messages)
                )

                return jsonify(response), 200, headers

            elif request_type == "feedback":
                bucket_name = "bucket"
                feedback_input = FeedbackInputSchema().load(request_data)

                feedback = {
                    "is_positive": feedback_input["is_positive"],
                    "feedback_text": feedback_input["feedback_text"],
                    "messages": feedback_input["messages"],
                }

                print("Not yet saving feedback...")
                # TODO FeedbackHandler.save_feedback(feedback)
                return jsonify({"message": "Feedback submitted successfully."})
            else:
                return jsonify({"error": "Invalid request_type field"}), 400, headers

        except ValidationError as err:
            return jsonify(err.messages), 400, headers
        except Exception as e:
            return jsonify({"error": str(e)}), 500, headers


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
