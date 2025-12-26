# Chat with SQL Database

A Streamlit-based application that allows you to interact with SQL databases using natural language queries powered by LangChain and Groq's LLaMA model.

## Features

- 🗣️ **Natural Language Queries**: Ask questions about your database in plain English
- 🗄️ **Multiple Database Support**:
  - SQLite (local database included)
  - MySQL (connect to your own MySQL database)
-  **AI-Powered**: Uses Groq's LLaMA 3.3 70B model for intelligent query generation
-  **Chat Interface**: Interactive chat-based UI with message history
-  **SQL Agent**: Automatically generates and executes SQL queries based on your questions

## Prerequisites

- Python 3.8 or higher
- Groq API key (get it from [Groq Console](https://console.groq.com))
- MySQL server (if connecting to MySQL database)

## Installation

1. Clone or download this repository

2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

1. **Start the application:**

```bash
streamlit run app.py
```

2. **Configure the database:**

   - Choose between SQLite (default) or MySQL from the sidebar
   - For SQLite: The app uses the included `studentdb.db` file
   - For MySQL: Enter your connection details (host, user, password, database)

3. **Enter your Groq API key** in the sidebar

4. **Start chatting** with your database using natural language queries!

## Example Queries

- "How many students are in the database?"
- "Show me all male students"
- "What is the average age of students?"
- "List all students born after 2004"
- "Who are the students with gmail addresses?"

## Configuration

### SQLite Database

The app includes a sample `studentdb.db` with a students table containing:

- id
- name
- email
- gender
- date_of_birth
- created_at

### MySQL Connection

To connect to your MySQL database, provide:

- **Host**: MySQL server hostname (e.g., localhost)
- **User**: MySQL username
- **Password**: MySQL password
- **Database**: Database name

## Environment Variables

You can optionally create a `.env` file to store your API key:

```
GROQ_API_KEY=your_api_key_here
```

## Troubleshooting

### MySQL Connection Issues

- Ensure MySQL server is running
- Verify connection credentials
- Check if the database exists
- Ensure the MySQL user has proper permissions

### API Key Issues

- Verify your Groq API key is valid
- Check your API usage limits

## Technologies Used

- **Streamlit**: Web application framework
- **LangChain**: Framework for building LLM applications
- **Groq**: Fast AI inference
- **SQLAlchemy**: SQL toolkit and ORM
- **MySQL Connector**: MySQL database driver

## Security Notes

- API keys are handled securely using Streamlit's password input
- Database passwords are not stored or logged
- SQLite database is opened in read-only mode by default

## License

This project is open source and available under the MIT License.

Creator is YoYo Ayan 