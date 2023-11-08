import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Sample data
tasks = [
    {"name": "Project A", "start": "2023-01-01", "end": "2023-01-05", "assignee": "Alice"},
    {"name": "Task A1", "start": "2023-01-01", "end": "2023-01-02", "assignee": "Alice"},
    {"name": "Task A2", "start": "2023-01-02", "end": "2023-01-03", "assignee": "Bob"},
    {"name": "Project B", "start": "2023-01-03", "end": "2023-01-08", "assignee": "Charlie"},
    {"name": "Task B1", "start": "2023-01-03", "end": "2023-01-05", "assignee": "Charlie"},
    {"name": "Task B2", "start": "2023-01-05", "end": "2023-01-06", "assignee": "David"},
]

fig, ax = plt.subplots(figsize=(10, 6))

# Set the date format for the x-axis
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

# Create the Gantt chart bars
for i, task in enumerate(tasks):
    start_date = datetime.strptime(task["start"], "%Y-%m-%d")
    end_date = datetime.strptime(task["end"], "%Y-%m-%d")
    ax.barh(task["name"] + " (" + task["assignee"] + ")", (end_date - start_date).days, left=start_date, align='center', color='skyblue', edgecolor='black')

# Invert y axis for chronological order
ax.invert_yaxis()

# Set the grid, title, and labels
ax.grid(axis='x', linestyle='--')
ax.set_title('Gantt Chart with Tasks, Dates, and Assignees')
ax.set_xlabel('Date')
ax.set_ylabel('Tasks and Assignees')

plt.tight_layout()
plt.show()
