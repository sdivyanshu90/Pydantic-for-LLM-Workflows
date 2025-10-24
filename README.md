# Pydantic for LLM Workflows

This repository contains personal notes, code exercises, and project work from the **[DeepLearning.AI](https://www.deeplearning.ai/)** short course, **[Pydantic for LLM Workflows](https://www.deeplearning.ai/short-courses/pydantic-for-llm-workflows/).**

## About This Course

Large Language Models (LLMs) naturally provide free-form text responses, which works well for unstructured generation, such as article summaries or brainstorming exercises. However, when you’re building an LLM into a larger software system—where you want to pass data from an LLM response to the next component in a predictable way—structured output becomes essential.

In this course, you’ll learn to move beyond free-form LLM responses and generate structured outputs that are easier to process and connect to other tools.

You’ll begin by understanding what structured output is and why it matters when building applications that use LLMs. Through the example of a customer support assistant, you’ll learn different methods of using Pydantic to ensure an LLM gives you the expected data and format you need. These methods ensure that the LLM’s responses are complete, correctly formatted, and ready to use, whether that means creating support tickets, triggering tools, or routing requests.

Throughout the course, you’ll gain core data validation skills that can be helpful in any software system you build, where you want to pass data from one component to the next. You’ll also learn how modern frameworks and LLM providers support structured outputs and function calls using Pydantic under the hood.

In detail, you’ll:

* Learn the basics of Pydantic, and practice different approaches for getting structured data from Pydantic models.

* Validate user input, catching issues like badly formatted emails or missing fields before they cause problems.

* Use Pydantic data models directly in your API calls to different LLM providers and agent frameworks as a reliable way to get a structured response.

* Combine structured outputs and tool-calling with Pydantic models in your application.

Pydantic is one of the most popular data validation frameworks out there. It sees over 300 million downloads a month, making it also one of the most popular Python packages, and that’s because data validation is at the core of any application.

By the end of the course, you’ll be able to build LLM-powered applications where every step is structured, validated, and ready to plug into your workflow.

## Course Topics

This section contains detailed notes and explanations for each major topic covered in the course.

<details>
<summary><strong>1. Introduction to Pydantic for LLM Workflows</strong></summary>

<h4>Introduction to Pydantic for LLM Workflows</h4>

The integration of Large Language Models (LLMs) into software applications has marked a significant paradigm shift. We've moved from strictly deterministic logic to probabilistic, generative systems. This shift unlocks incredible capabilities—natural language interfaces, content generation, and complex reasoning—but it also introduces a fundamental challenge: unpredictability. LLMs are trained to generate human-like text, not machine-readable data. This "unstructured output" problem is the primary bottleneck in building robust, production-grade applications on top of LLMs.

An "LLM workflow" refers to any process where an LLM is a component in a larger system. This could be as simple as a chatbot answering a question or as complex as an automated agent that reads emails, extracts tasks, schedules them in a calendar, and notifies a user. In any ofthese workflows, the data flows between components. The LLM's output must be consumed by another piece of software—a database, an API, a frontend application, or another function.

This is where the problem becomes acute. Imagine a customer support bot designed to create a support ticket. You prompt the LLM: "The user 'john.doe@example.com' is reporting that his 'premium subscription' is not working. The issue is 'cannot access dashboard' and his priority is 'high'."

An LLM might respond in countless ways:

Good, but unstructured: "Okay, I've logged a high-priority ticket for John Doe (john.doe@example.com) regarding the premium subscription."

Missing data: "Ticket created for `john.doe@example.com`. Issue: 'cannot access dashboard'." (Priority is missing).

Incorrect format: "User: John, Email: `john.doe@example.com`, Priority: 3 (High)" (Priority is a string 3 (High) instead of an expected int or enum).

Conversational padding: "I'm sorry to hear that! I will create a ticket right away. Here are the details: \n - User: `john.doe@example.com` \n - Priority: high \n - Issue: cannot access dashboard"

A traditional software component expecting a clean JSON object like `{"email": "...", "priority_level": 3, "summary": "..."}` would fail in all of these cases. The application logic would require complex, brittle, and unreliable regular expressions (regex) or string parsing to "find" the data it needs. This is a maintenance nightmare.

Enter Pydantic.

Pydantic is a Python library for data validation and settings management using Python type hints. At its core, it provides a way to define a "schema" for your data as a simple Python class. It enforces that any data claiming to match that schema actually does.

The "Pydantic for LLM Workflows" concept is about using Pydantic as the rigid, deterministic contract that bridges the gap between the probabilistic, unstructured LLM and the deterministic, structured application code.

Instead of asking the LLM to just "write a response," you instruct it to "fill out this form." Pydantic is that form.

This workflow fundamentally changes the application architecture:

Define Your "Form" (The Schema): You define a Pydantic BaseModel that represents the exact data structure you need.
```python
from pydantic import BaseModel, EmailStr
from typing import Literal

class SupportTicket(BaseModel):
    email: EmailStr
    subscription_level: Literal["free", "basic", "premium"]
    priority: Literal["low", "medium", "high"]
    summary: str
```

This class is now the "single source of truth." It clearly states: "I need an email, a subscription level (which must be one of three values), a priority (also a specific enum), and a summary string."

Instruct the LLM (Prompt Engineering): You modify your prompt to include instructions for the LLM to provide its answer in a JSON format that matches this schema. Modern LLM APIs (like OpenAI's, Anthropic's, and Google's) have a "tool-calling" or "structured output" feature that allows you to pass this schema directly to the model, compelling it to generate a matching JSON.

Parse and Validate (The "Pydantic" Step): The LLM now returns a JSON string:
`'{"email": "john.doe@example.com", "subscription_level": "premium", "priority": "high", "summary": "cannot access dashboard"}'`

Your application code does not trust this output. It validates it:
```python
llm_json_output = '...'
try:
    ticket = SupportTicket.model_validate_json(llm_json_output)
    # Now `ticket` is a guaranteed-to-be-valid SupportTicket object
    # You can access ticket.email, ticket.priority, etc. with
    # full type safety and IDE autocompletion.
    create_ticket_in_db(ticket)
except ValidationError as e:
    # The LLM failed to follow instructions
    print(f"LLM output was invalid: {e}")
    # This is where the workflow gets smart...
```

The Feedback Loop (The "Workflow" Step): What if the LLM output was {"email": "not-an-email", "priority": "urgent"}? Pydantic's ValidationError would be incredibly specific:

email: "Input should be a valid email address"

priority: "Input should be 'low', 'medium', or 'high' (was 'urgent')"

Instead of the application crashing, you can programmatically catch this error, format it, and send it back to the LLM in a "retry" prompt:
"Your previous output was invalid. Please correct the following errors and try again: \n - For the 'email' field: Input should be a valid email address. \n - For the 'priority' field: Input should be 'low', 'medium', or 'high' (you provided 'urgent'). \n Please provide the corrected, full JSON object."

This "Define -> Instruct -> Validate -> Feedback" loop is the essence of a robust LLM workflow. Pydantic is the engine that powers the "Define" and "Validate" steps.

The benefits are transformative:

* Reliability: Your application is protected from malformed, incomplete, or invalid data from the LLM.

* Type Safety: Downstream code (like create_ticket_in_db) can trust that ticket.priority is a valid string, not an integer or a misspelled value.

* Developer Experience: You get IDE autocompletion (ticket. shows email, priority, etc.). The schema (SupportTicket) is clean, readable Python, not a complex external JSON schema file.

* Maintainability: If you need to add a user_id field, you add user_id: int to the Pydantic model. The entire system—from the prompt instructions to the validation—updates accordingly.

* Smart Prompting: You can use Field(description=...) in your Pydantic model to give hints to the LLM on how to fill out a specific field, effectively embedding your prompt engineering inside your data model.

In summary, this introductory topic establishes the core problem of modern LLM development—the "structured data" gap. It positions Pydantic not just as a data validation tool, but as the fundamental "contract" that enables reliable communication between the probabilistic world of LLMs and the deterministic world of software.

</details>

<details>
<summary><strong>2. Pydantic Model Basics</strong></summary>

<h4>Pydantic Model Basics</h4>

At the heart of Pydantic is the BaseModel. It's the class you inherit from to create your own "schema" or data model. This topic lays the foundation for all other Pydantic-powered workflows by exploring how to define, customize, and use these models.

1. Defining a Basic Model
You define a model by creating a class that inherits from pydantic.BaseModel. The "schema" is defined using standard Python type hints for class attributes.
```python
from pydantic import BaseModel

class User(BaseModel):
    id: int
    name: str
    is_active: bool = True  # A field with a default value
```

This simple class already provides a wealth of functionality:

Instantiation: You can create an instance just like a normal Python class:
user = User(id=1, name="Alice")
(Note: is_active will be True by default).

Type Coercion: If you pass data of a "close" type, Pydantic will try to coerce it.
user = User(id="1", name="Alice")
Here, id="1" (a string) will be automatically coerced into id=1 (an integer). This is extremely useful for data coming from web requests or LLMs, which might send all values as strings.

Validation: If coercion fails or the type is wrong, it raises a ValidationError.
User(id="one", name="Alice") -> ValidationError (cannot coerce "one" to an int)
User(id=1) -> ValidationError (the name field is required and missing)

2. Parsing and Serializing Data
This is the most common use case. You have data (e.g., from an API, a database, or an LLM) and you want to parse it into your model.

model_validate(data): (Replaces parse_obj in Pydantic v1)
This method takes a Python dictionary and validates it.
```python
data_dict = {"id": 2, "name": "Bob", "is_active": False}
bob = User.model_validate(data_dict)
# bob is now a User(id=2, name='Bob', is_active=False)
```

model_validate_json(json_data):
This method takes a JSON string, first parses it into a Python dict, and then validates it. This is the most common method for LLM outputs.
```python
json_str = '{"id": 3, "name": "Charlie"}'
charlie = User.model_validate_json(json_str)
# charlie is User(id=3, name='Charlie', is_active=True)
```

model_dump(): (Replaces .dict() in v1)
This converts a model instance back into a Python dictionary.
charlie.model_dump() -> {'id': 3, 'name': 'Charlie', 'is_active': True}

model_dump_json(): (Replaces .json() in v1)
This converts the instance directly into a JSON string.
charlie.model_dump_json() -> '{"id":3,"name":"Charlie","is_active":true}'

3. Advanced Field Types
Pydantic goes far beyond basic types like int and str. It leverages Python's typing module and provides its own set of "strict" types.

Standard Library Types:

* List[str]: A list of strings.

* Dict[str, int]: A dictionary with string keys and integer values.

* Optional[int] = None: An integer field that is allowed to be None. (Replaced by int | None = None in Python 3.10+).

* Union[int, str]: A field that can be either an integer or a string.

* Literal["low", "medium", "high"]: An enum-like field that must be one of those exact string values. Incredibly useful for LLM workflows.

Pydantic's Special Types:

* EmailStr: Validates that a string is a valid email format.

* HttpUrl: Validates that a string is a valid URL.

* PositiveInt: An integer that must be > 0.

* NegativeFloat: A float that must be < 0.

* constr(min_length=5, max_length=50, pattern=r'^[A-Z]'): A constrained string that must be between 5 and 50 chars and start with a capital letter.

4. Nested Models <br>
This is how you build complex, hierarchical data structures. You simply use another BaseModel as a type hint.
```python
class Address(BaseModel):
    street: str
    city: str
    zip_code: str

class User(BaseModel):
    id: int
    name: str
    address: Address  # Nesting the Address model


# Now, Pydantic's validator will recursively validate the data.

data = {
    "id": 1,
    "name": "Alice",
    "address": {
        "street": "123 Main St",
        "city": "Anytown",
        "zip_code": "12345"
        # "country": "USA" # This would cause an error if extra='forbid'
    }
}
user = User.model_validate(data)
print(user.address.city)  # Output: Anytown
```

5. Customizing Fields with Field <br>
The Field function allows you to add extra configuration to a field, which is essential for LLM workflows.
```python
from pydantic import BaseModel, Field

class User(BaseModel):
    id: int = Field(
        ...,  # The '...' means this field is required
        gt=0,  # "Greater Than" 0 (a validator)
        description="The user's unique identifier."
    )
    name: str = Field(
        max_length=100,
        description="The user's full name."
    )
    # An alias is used for mapping to data with different key names
    email: EmailStr = Field(
        alias="userEmail",
        description="The user's primary email address."
    )
```

Validation: gt=0, max_length=100 add validation rules.

Alias: alias="userEmail" tells Pydantic that when parsing data, it should look for the key "userEmail" and map it to the email attribute.

Description: description="..." is the most important part for LLM workflows. When you auto-generate a schema to send to an LLM, these descriptions are included as instructions for the LLM on how to fill in that field.

6. Custom Validators and Computed Fields <br>

@field_validator: You can write your own functions to validate or transform data.
```python
from pydantic import field_validator

class User(BaseModel):
    username: str

    @field_validator("username")
    @classmethod
    def clean_username(cls, v: str) -> str:
        # A 'before' validator that runs on the raw input
        if not v.isalnum():
            raise ValueError("Username must be alphanumeric")
        return v.lower()  # Normalizes the data
```

@computed_field: You can create new fields that are derived from other fields.
```python
from pydantic import computed_field

class Name(BaseModel):
    first: str
    last: str

    @computed_field
    @property
    def full(self) -> str:
        return f"{self.first} {self.last}"

n = Name(first="John", last="Doe")
print(n.full) # Output: John Doe
print(n.model_dump()) # {'first': 'John', 'last': 'Doe', 'full': 'John Doe'}
```

7. Model Configuration (ConfigDict)
You can change the behavior of a model using an inner ConfigDict.
```python
from pydantic import ConfigDict

class MyModel(BaseModel):
    model_config = ConfigDict(
        extra="ignore",  # Ignore extra fields in the input data
        # extra="forbid", # (Default) Fail if extra fields are present
        from_attributes=True # (was orm_mode) Allow loading from other classes
    )
    name: str
```

extra="ignore" is very common when dealing with verbose API responses where you only care about a few fields.

Understanding these basics—BaseModel, parsing/dumping, nested models, and Field customization—provides the complete toolbox needed to define any data structure you want to extract from an LLM.

</details>

<details>
<summary><strong>3. Validating LLM Responses</strong></summary>

<h4>Validating LLM Responses</h4>

This topic is the practical application of the first two. You have an LLM, you have a Pydantic model, and now you're trying to make them work together. "Validating LLM responses" is the process of building a resilient system that can handle the fact that LLM outputs are not 100% reliable. This process can be broken down into three phases: Pre-processing, Validation, and The Retry Loop.

Phase 1: Pre-processing (Cleaning the Raw Output)
LLMs, especially models not explicitly trained for JSON output, will often "wrap" their structured data in conversational text.

A common-but-failed output might look like this:

"Sure, I can help with that! Here is the JSON data you requested:
```json
{
    "name": "John Doe",
    "age": "thirty", // This is a mistake
    "email": "john.doe@example.com"
}
```

I hope this helps!"

If you pass this entire string to `MyModel.model_validate_json()`, it will fail instantly with a `JSONDecodeError` because the string doesn't start with `{`.

The pre-processing step involves "sanitizing" this raw text to extract the *actual* JSON payload.
* **Regex Extraction:** The most common method is to use a regular expression to find content within JSON code blocks.
    ```python
    import re
    
    raw_output = "..." # The text above
    match = re.search(r"```json\n(.*?)\n```", raw_output, re.DOTALL)
    
    if match:
        json_str = match.group(1)
    else:
        # Fallback: maybe it's just the JSON?
        # Or maybe it's a lost cause.
        json_str = raw_output 
    ```
* **JSON "Healing":** Sometimes the LLM produces *almost* valid JSON, but with syntax errors like trailing commas, missing quotes, or using single quotes.
    `{'name': 'John', 'age': 30,}`
    This is technically invalid. Libraries like `json_repair` or `dirtyjson` are designed to fix these common syntax errors before parsing.
    ```python
    from json_repair import repair_json
    
    broken_json = "{'name': 'John', 'age': 30,}"
    fixed_json = repair_json(broken_json)
    # fixed_json is now '{"name": "John", "age": 30}'
    ```
Only after this pre-processing step do you have a string that is *ready* for Pydantic validation.

**Phase 2: The `try...except` Validation Block** <br>
This is the core of the validation workflow. You *never* assume the pre-processed JSON string is valid. You *always* wrap the Pydantic parsing call in a `try...except` block.

```python
from pydantic import BaseModel, ValidationError

class User(BaseModel):
    name: str
    age: int
    email: EmailStr

# Assume json_str comes from Phase 1
json_str = '{"name": "John Doe", "age": "thirty", "email": "john.doe@example.com"}'

try:
    user = User.model_validate_json(json_str)
    # --- SUCCESS ---
    # If we get here, `user` is a valid User object.
    print("Validation successful!")
    print(user)
    # proceed_with_workflow(user)

except ValidationError as e:
    # --- FAILURE ---
    # This is the crucial part. Pydantic failed.
    # `e` is an object containing rich error details.
    print("Validation FAILED!")
    # We don't just give up; we inspect the error.
    error_details = e.errors() 
```

The e.errors() method is the key. It returns a list of dictionaries, with each dictionary detailing a specific failure. For the json_str above, e.errors() would look like this:
```json
[
  {
    "type": "int_parsing",
    "loc": ["age"],
    "msg": "Input should be a valid integer, unable to parse string as an integer",
    "input": "thirty"
  }
]
```

This is machine-readable feedback. It tells you:

* loc: ["age"]: The error happened at the age field.

* msg: A human-readable error message.

* type: A programmatic error type.

* input: The exact bad value that caused the failure.

**Phase 3: The Validation Feedback Loop** <br>
Now, what do you do with this error_details? You use it to ask the LLM to fix its own mistake. This is the "retry loop," and it's what makes an LLM workflow robust.

You build a new prompt that includes the original query, the LLM's broken response, and the specific validation errors.

`(Inside the `except ValidationError as e:` block)`

1. Format the errors from Pydantic into a human-readable string
error_feedback = []
for error in e.errors():
    field = ".".join(map(str, error['loc']))
    message = error['msg']
    error_feedback.append(f"Error in field '{field}': {message}. You provided: {error['input']}")

feedback_prompt = "\n".join(error_feedback)
feedback_prompt is now:
"Error in field 'age': Input should be a valid integer... You provided: thirty"

2. Build a new prompt for the LLM
original_prompt = "Extract user info from '...'" # The very first prompt

retry_prompt = f"""
I asked you to perform a task: {original_prompt}

You provided the following JSON response:
{json_str}

However, this response failed validation with the following errors:
{feedback_prompt}

Please correct these errors and provide the full, valid JSON object again.
"""

3. Call the LLM *again* with this new prompt
`new_llm_output = llm_api_call(retry_prompt)`
... and then you run new_llm_output through the whole
Phase 1 -> Phase 2 validation again.


This retry loop can be run 2-3 times. If it still fails, you can then escalate to a human or return a final error.

This "self-healing" or "self-correcting" loop is a critical pattern. It transforms Pydantic from a simple "pass/fail" gate into an active participant in the generation process.

Tooling that automates this:
This pattern is so common and powerful that libraries have been built to abstract it away.

Instructor: The instructor.patch() client does this retry loop automatically. When you make a call with response_model=User, if the first validation fails, instructor will automatically catch the ValidationError, build the retry prompt, and call the LLM again, all behind the scenes.

LangChain: LangChain's PydanticOutputParser is the component that does the validation. It provides a parse(llm_output) method that throws the ValidationError. It also has a get_format_instructions() method to help prevent the error in the first place. You would then manually build the retry loop, or use a more advanced agent chain like create_structured_output_runnable which can incorporate retries.

In conclusion, "validating LLM responses" is not a single step but a process. It starts with cleaning the raw text, moves to a try...except block for Pydantic parsing, and culminates in a powerful retry loop that uses Pydantic's rich ValidationError data to tell the LLM how to fix its own mistakes.

</details>

<details>
<summary><strong>4. Passing a Pydantic Model in Your API Call</strong></summary>

<h4>Passing a Pydantic Model in Your API Call</h4>

This topic represents a shift from reactive validation (fixing bad output, as in Topic 3) to proactive generation (forcing good output from the start). Instead of just hoping the LLM returns JSON that matches your model, you pass the Pydantic model's schema to the LLM as part of the API call itself.

This compels the LLM to generate a response that structurally conforms to your model. There are two primary ways to do this: manually via prompt engineering, and automatically via native "Tool Calling" APIs.

Method 1: Manual Prompt Engineering (The "Format Instructions" Method)
This method works with any LLM, even older or less capable ones that don't have a special "tool calling" feature. The idea is to serialize your Pydantic model's schema into text and include it directly in your prompt.

Define your Pydantic Model:
```python
from pydantic import BaseModel, Field

class User(BaseModel):
    name: str = Field(..., description="The user's full name.")
    age: int = Field(..., description="The user's age in years.")
```

Get the Schema: Pydantic models can generate a JSON Schema, which is a standardized, language-agnostic way to describe a JSON data structure.

`schema_json_string = User.model_json_schema_dump(indent=2)`
schema_json_string is now:
```json
{
  "properties": {
    "name": {
      "description": "The user's full name.",
      "title": "Name",
      "type": "string"
    },
    "age": {
      "description": "The user's age in years.",
      "title": "Age",
      "type": "integer"
    }
  },
  "required": ["name", "age"],
  "title": "User",
  "type": "object"
}
```


Notice how the description from Field is included! This is a prompt for the LLM.

Inject into the Prompt: You create a prompt that tells the LLM to provide its answer only in a JSON format matching this schema.
```python
prompt = f"""
Extract the user's name and age from the following text.

Text: "John Doe is 30 years old."

You MUST format your response as a valid JSON object.
Do not include any other text, just the JSON.
The JSON object must conform to the following JSON Schema:

{schema_json_string}
"""
```

Execute and Validate:
The LLM will (hopefully) respond with:
`'{"name": "John Doe", "age": 30}'`
...which you then validate using User.model_validate_json().

Tooling: LangChain's PydanticOutputParser automates this. Its get_format_instructions() method generates a text-based description of the model (often more concise than the full JSON schema) specifically designed to be put in a prompt.

Downside: This is still just "prompt engineering." The LLM is following instructions to produce text. It might still fail, add conversational padding ("Here is the JSON: ..."), or produce a malformed JSON string. This method still requires the full "Validate and Retry Loop" from Topic 3.

Method 2: Native Tool/Function Calling (The "Structured Output" Method)
This is the modern, preferred, and far more reliable method. Modern LLM providers (OpenAI, Anthropic, Google) have updated their APIs to accept a list of tools. The LLM can then be forced to respond by "calling" one of these tools.

This "tool" is, in effect, your Pydantic model.

Define your Model: Same as before.
```python
from pydantic import BaseModel, Field

class User(BaseModel):
    name: str = Field(..., description="The user's full name.")
    age: int = Field(..., description="The user's age in years.")
```

Get the JSON Schema: Same as before.
`user_schema = User.model_json_schema()`

Format for the API: You format this schema into the API's specific tools format.
(This example uses OpenAI's format)
```json
tools_payload = [
    {
        "type": "function",
        "function": {
            "name": "UserInfo",
            "description": "Extracts user information",
            # You insert the Pydantic schema here
            "parameters": user_schema 
        }
    }
]
```


Make the API Call:
You make the API call, passing the tools payload. Crucially, you also set tool_choice to force the LLM to use your tool.

Example using OpenAI's client
```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "John Doe is 30 years old."}],
    tools=tools_payload,
    tool_choice={"type": "function", "function": {"name": "UserInfo"}}
)
```

Parse the Response:
The LLM's response will not be text. It will be a special tool_calls object.
```json
{
  "role": "assistant",
  "content": null,
  "tool_calls": [
    {
      "id": "call_abc123",
      "type": "function",
      "function": {
        "name": "UserInfo",
        "arguments": "{\"name\":\"John Doe\",\"age\":30}"
      }
    }
  ]
}
```


The magic is in response.choices[0].message.tool_calls[0].function.arguments.
This arguments field is a JSON string that the LLM guarantees will successfully parse according to the schema you provided.

Validate (The Final Step):

tool_call = response.choices[0].message.tool_calls[0]
arguments_str = tool_call.function.arguments

* This step is *still* Pydantic
* It's now parsing the *guaranteed-to-be-valid-JSON*
* into a real Python object.
```python
user = User.model_validate_json(arguments_str)
```

* user is User(name='John Doe', age=30)


This method is vastly superior because:

It's guaranteed to be valid JSON. No more regex or JSON healing.

It's guaranteed to match the schema's structure (keys,
data types).

It eliminates conversational padding.

It's more reliable and faster.

Note: It doesn't guarantee the data is logical (it might still hallucinate age: 999), which is why you still need Pydantic's data validators (e.g., Field(gt=0, lt=120)).

Tooling (The "Abstraction" Method):
This native method is so powerful but verbose that libraries have abstracted it.

Instructor: This is instructor's entire purpose. It does all of Method 2's steps (get schema, format tools, force tool_choice, parse arguments) in one line.
```python
import instructor

1. Patch the client
client = instructor.patch(openai.OpenAI())

2. Make the call with `response_model`
user = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "John Doe is 30 years old."}],
    response_model=User  # This is the magic
)
```
`user` is *already* a validated Pydantic User object.
user is `User(name='John Doe', age=30)`


LangChain: `llm.with_structured_output(User)` achieves the same thing. It wraps the LLM and handles the schema formatting and response parsing.

In summary, "passing a Pydantic model" is the key to forcing structured output. You can do it manually by "prompting" with the schema, or—far more effectively—by using the tools parameter of modern LLM APIs. Libraries like instructor make this second method trivial by adding a response_model parameter to the API call.

</details>

<details>
<summary><strong>5. Tool Calling</strong></summary>

<h4>Tool Calling</h4>

"Tool calling" (also known as "function calling") is the mechanism that allows a Large Language Model (LLM) to "take actions" in the real world. It's the most critical concept for building agents, RAG (Retrieval-Augmented Generation) systems, and any application where the LLM needs to interact with external data or services.

At its core, tool calling is an orchestration loop where the LLM pauses its text generation and requests that the developer's application code be run. Pydantic's role in this loop is to be the unambiguous, validated "contract" for all data that flows between the LLM and the tools.

The Tool Calling Loop (Step-by-Step):

Imagine a user asks, "What's the weather in Boston and how many unread emails do I have?"
This requires two tools: get_weather and get_email_count.

Pydantic's First Role: Defining the Tool's Input Schema
First, you define your tools as Python functions. Then, you define Pydantic models for their arguments.

from pydantic import BaseModel, Field

Schema for the get_weather tool
```python
class GetWeatherArgs(BaseModel):
    location: str = Field(..., description="The city and state, e.g., 'San Francisco, CA'")

# Schema for the get_email_count tool
class GetEmailArgs(BaseModel):
    folder: str = Field("inbox", description="The email folder to check, e.g., 'inbox' or 'spam'")

# The actual tools
def get_weather(location: str) -> dict:
    # Code to call a weather API...
    return {"temperature": 72, "unit": "F", "conditions": "Sunny"}

def get_email_count(folder: str) -> dict:
    # Code to call a Gmail API...
    return {"unread_count": 5}
```

Pydantic models are perfect for this because their JSON Schema (with field descriptions) provides the exact instructions the LLM needs to understand what arguments to provide.

Step 1: The User's Query (The First LLM Call)
You send the user's prompt to the LLM, along with the schemas of all available tools.

messages = [{"role": "user", "content": "What's the weather in Boston and how many unread emails do I have?"}]

tools = [

{ "type": "function", "function": { "name": "get_weather", "parameters": GetWeatherArgs.model_json_schema() } }

{ "type": "function", "function": { "name": "get_email_count", "parameters": GetEmailArgs.model_json_schema() } }
]

You make the API call: `client.chat.completions.create(model="gpt-4o", messages=messages, tools=tools)`

Step 2: The LLM's Response (A Request to Call Tools)
The LLM does not answer the user. Instead, it analyzes the prompt, sees that it needs external data, and returns a tool_calls object requesting to run both tools.
```json
{
  "role": "assistant",
  "content": null,
  "tool_calls": [
    {
      "id": "call_abc",
      "type": "function",
      "function": {
        "name": "get_weather",
        "arguments": "{\"location\":\"Boston, MA\"}" 
      }
    },
    {
      "id": "call_xyz",
      "type": "function",
      "function": {
        "name": "get_email_count",
        "arguments": "{\"folder\":\"inbox\"}"
      }
    }
  ]
}
```

The LLM has correctly identified the tools and generated the arguments for each one, using the descriptions (e.g., "The city and state..."). It also inferred "inbox" as the default folder.

Step 3: The Developer's Code (Validation and Execution)
This is where your application code takes over, and Pydantic's second role becomes critical. You must loop through the tool_calls, validate the arguments, and run the functions.

1. Get the assistant's response (the tool_calls object)
response_message = response.choices[0].message

2. Append this message to our message history
messages.append(response_message) 

3. Loop through each tool call
```python
for tool_call in response_message.tool_calls:
    function_name = tool_call.function.name
    arguments_str = tool_call.function.arguments

    # **PYDANTIC'S SECOND ROLE: Validating the LLM's Arguments**
    try:
        if function_name == "get_weather":
            args = GetWeatherArgs.model_validate_json(arguments_str)
            tool_result = get_weather(location=args.location)
        
        elif function_name == "get_email_count":
            args = GetEmailArgs.model_validate_json(arguments_str)
            tool_result = get_email_count(folder=args.folder)

        # 4. Append the *result* to the message history
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": json.dumps(tool_result) # Result must be a string
            }
        )
    except ValidationError as e:
        # Handle cases where the LLM sent bad arguments
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": f"Error validating arguments: {e}"
            }
        )
```

Pydantic protects your functions (get_weather, get_email_count) from receiving invalid data from the LLM. If the LLM sent {"loc": "Boston"}, GetWeatherArgs would raise a ValidationError because the location field is missing.

Step 4: The Final LLM Call (Synthesizing the Answer)
Your messages list now looks like this:

role: user, content: "What's the weather..."

role: assistant, tool_calls: [...] (The request to run tools)

role: tool, tool_call_id: "call_abc", content: '{"temperature": 72, ...}' (Weather result)

role: tool, tool_call_id: "call_xyz", content: '{"unread_count": 5}' (Email result)

You make a second API call, sending this entire history.
final_response = client.chat.completions.create(model="gpt-4o", messages=messages, tools=tools)

Step 5: The LLM's Final Response (A Natural Language Answer)
Now that the LLM has the user's question and the data from the tools, it will synthesize a final answer.
final_response.choices[0].message.content will be:
"The weather in Boston is 72°F and sunny. You also have 5 unread emails in your inbox."

Pydantic's Third (Optional) Role: Validating Tool Output
You can also define Pydantic models for the return values of your tools.
class WeatherData(BaseModel): temperature: int, unit: Literal['F', 'C']
After tool_result = get_weather(...), you could run WeatherData.model_validate(tool_result). This ensures the data you pass back to the LLM (in the tool message) is also clean and valid, preventing errors if your own API (e.g., the weather API) returns unexpected data.

Tooling:
Libraries like LangChain and Instructor are primarily designed to manage this complex, multi-step loop.

Instructor: You can define a tool as a Python function, and instructor will automatically wrap its type hints (location: str) in a Pydantic model, generate the schema, and manage the validation step.

LangChain: LangChain's "Agents" (like the OpenAI Tools agent) are pre-built executors for this exact loop. You just provide the Python functions (decorated with @tool), and the agent handles all five steps: calling the LLM, parsing tool calls, validating, executing, and sending the results back for the final answer.

In summary, "tool calling" is the fundamental architecture for building capable LLM applications. Pydantic is the "glue" that holds this architecture together, providing the robust, type-safe, and self-documenting "contract" for all data that moves between the LLM's brain and the application's tools.

</details>

## Acknowledgement

This repository is for educational and personal learning purposes only. All course materials, content, and intellectual property are owned by **[DeepLearning.AI](https://www.deeplearning.ai)**.

A huge thank you to DeepLearning.AI for creating this invaluable and practical course on leveraging Pydantic for modern, robust, and reliable LLM-powered applications.