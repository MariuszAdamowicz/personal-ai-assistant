from src.my_agent.tools.tool import Tool


class WebSearchTool(Tool):

    def __init__(self, tavily_client):
        self.tavily_client = tavily_client

    def description(self):
        return {
              "type": "function",
              "function": {
                "name": "web_search",
                "description": "Search information in the web",
                "parameters": {
                  "type": "object",
                  "properties": {
                    "question": {
                      "type": "string",
                      "description": "Question, e.g. Who is Messi?"
                    }
                  },
                  "required": ["question"]
                }
              }
            }

    def base_context(self):
        return "Use the web_search function to retrieve live data from the internet."

    def call(self, parameters):
        question = parameters.get("question")
        if not question:
            return {"error": "Missing 'question' parameter."}
        return self.tavily_client.search(question)
