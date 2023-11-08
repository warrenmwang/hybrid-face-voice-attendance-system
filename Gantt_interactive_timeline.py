You: import plotly.express as px
import pandas as pd

# Sample data with tasks and subtasks
df = pd.DataFrame([
    dict(Task="Project A", Start='2023-01-01', End='2023-01-05', Resource="Task"),
    dict(Task="Task A1", Start='2023-01-01', End='2023-01-02', Resource="Subtask"),
    dict(Task="Task A2", Start='2023-01-02', End='2023-01-03', Resource="Subtask"),
    dict(Task="Project B", Start='2023-01-03', End='2023-01-08', Resource="Task"),
    dict(Task="Task B1", Start='2023-01-03', End='2023-01-05', Resource="Subtask"),
    dict(Task="Task B2", Start='2023-01-05', End='2023-01-06', Resource="Subtask"),
])

# Create the Gantt chart
fig = px.timeline(df, x_start="Start", x_end="End", y="Task", color="Resource", title="Gantt Chart with Tasks and Subtasks")
fig.update_yaxes(categoryorder="total ascending")

# Show the chart
fig.show()
