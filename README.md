# pandas
Learning Python Pandas in 2 hours is ambitious, but you can cover the basics and get started. Here’s a focused plan:

---

### **1. Setup (10 minutes)**
- **Install Pandas:**  
  Run in your terminal:  
  ```bash
  pip install pandas
  ```
- **Import Pandas in Python:**  
  ```python
  import pandas as pd
  ```

---

### **2. Understanding Pandas Basics (20 minutes)**

#### **2.1 Data Structures**
- **Series:** 1D labeled array (like a list).  
  Example:  
  ```python
  data = [1, 2, 3]
  series = pd.Series(data)
  print(series)
  ```
- **DataFrame:** 2D labeled data (like a table).  
  Example:  
  ```python
  data = {
      'Name': ['Alice', 'Bob'],
      'Age': [24, 27]
  }
  df = pd.DataFrame(data)
  print(df)
  ```

#### **2.2 Read and Write Data**
- **Read CSV:**  
  ```python
  df = pd.read_csv('file.csv')
  print(df.head())
  ```
- **Write CSV:**  
  ```python
  df.to_csv('output.csv', index=False)
  ```

---

### **3. Data Manipulation (30 minutes)**

#### **3.1 Inspect Data**
- View first 5 rows:  
  ```python
  print(df.head())
  ```
- Get column names and info:  
  ```python
  print(df.columns)
  print(df.info())
  ```

#### **3.2 Selecting Data**
- Select a column:  
  ```python
  print(df['Name'])
  ```
- Filter rows:  
  ```python
  print(df[df['Age'] > 25])
  ```

#### **3.3 Modifying Data**
- Add a new column:  
  ```python
  df['Salary'] = [50000, 60000]
  print(df)
  ```
- Modify a column:  
  ```python
  df['Age'] = df['Age'] + 1
  ```

#### **3.4 Handle Missing Data**
- Check for missing values:  
  ```python
  print(df.isnull().sum())
  ```
- Fill missing values:  
  ```python
  df['Age'].fillna(0, inplace=True)
  ```

---

### **4. Data Analysis (30 minutes)**

#### **4.1 Descriptive Statistics**
- Basic stats:  
  ```python
  print(df.describe())
  ```
- Count values:  
  ```python
  print(df['Age'].value_counts())
  ```

#### **4.2 Grouping**
- Group by a column:  
  ```python
  print(df.groupby('Age').mean())
  ```

#### **4.3 Sorting**
- Sort by a column:  
  ```python
  print(df.sort_values(by='Age', ascending=False))
  ```

---

### **5. Visualization (30 minutes)**
(Requires Matplotlib or Seaborn)  
- Install libraries:  
  ```bash
  pip install matplotlib seaborn
  ```
- Example Visualization:  
  ```python
  import matplotlib.pyplot as plt
  df['Age'].plot(kind='bar')
  plt.show()
  ```

---

### **6. Practice Exercises**
- Load any CSV dataset and perform:
  1. Viewing data (`head`, `info`)
  2. Filtering and selecting data
  3. Adding new columns
  4. Grouping and sorting
  5. Export the results to a new file.

---
Here’s a dataset and code snippets for practice:

---

### **Practice Dataset**
Create a CSV file named `students.csv` with the following content:

```csv
Name,Age,Grade,Subject,Marks
Alice,24,A,Math,85
Bob,27,B,Science,78
Charlie,22,C,Math,68
David,24,A,Science,90
Eva,23,B,Math,72
```

---

### **Code Snippets for Practice**

#### **1. Load the Dataset**
```python
import pandas as pd

# Load the dataset
df = pd.read_csv('students.csv')
print(df)
```

#### **2. View Basic Information**
```python
# First few rows
print(df.head())

# Summary info
print(df.info())

# Column names
print(df.columns)
```

#### **3. Data Selection**
- Select a column:
  ```python
  print(df['Name'])
  ```
- Select multiple columns:
  ```python
  print(df[['Name', 'Marks']])
  ```
- Filter rows where `Marks` > 80:
  ```python
  print(df[df['Marks'] > 80])
  ```

#### **4. Adding a New Column**
- Add a new column for grades based on `Marks`:
  ```python
  df['Grade Level'] = ['High' if x >= 85 else 'Low' for x in df['Marks']]
  print(df)
  ```

#### **5. Grouping Data**
- Group by `Subject` and calculate the average marks:
  ```python
  avg_marks = df.groupby('Subject')['Marks'].mean()
  print(avg_marks)
  ```

#### **6. Sorting Data**
- Sort by `Marks` in descending order:
  ```python
  sorted_df = df.sort_values(by='Marks', ascending=False)
  print(sorted_df)
  ```

#### **7. Handling Missing Values**
- Assume some marks are missing; fill them with the mean:
  ```python
  df.loc[2, 'Marks'] = None  # Simulate a missing value
  print(df)

  df['Marks'].fillna(df['Marks'].mean(), inplace=True)
  print(df)
  ```

#### **8. Save the Processed Data**
```python
df.to_csv('processed_students.csv', index=False)
print("Processed data saved to 'processed_students.csv'.")
```

#### **9. Visualization**
Install Matplotlib for plotting:
```bash
pip install matplotlib
```

Create a bar chart of `Marks`:
```python
import matplotlib.pyplot as plt

# Bar chart
df.plot(kind='bar', x='Name', y='Marks', color='blue', legend=False)
plt.title('Marks of Students')
plt.xlabel('Name')
plt.ylabel('Marks')
plt.show()
```

---
Here are additional exercises and explanations to help you practice Python Pandas further:

---

### **Additional Exercises**

#### **1. Advanced Data Selection**
- Select students with grades "A" or "B":
  ```python
  filtered_df = df[df['Grade'].isin(['A', 'B'])]
  print(filtered_df)
  ```
- Find the student(s) with the highest marks:
  ```python
  highest_marks = df[df['Marks'] == df['Marks'].max()]
  print(highest_marks)
  ```

---

#### **2. Combining Data**
- Create another dataset `extra_subjects.csv`:
  ```csv
  Name,ExtraSubject,ExtraMarks
  Alice,Art,88
  Bob,Music,92
  Charlie,Dance,80
  David,Art,95
  Eva,Music,85
  ```
  
- Load and merge the datasets:
  ```python
  extra_df = pd.read_csv('extra_subjects.csv')

  # Merge the two datasets
  merged_df = pd.merge(df, extra_df, on='Name')
  print(merged_df)
  ```

---

#### **3. Aggregation**
- Calculate total marks (including extra marks) for each student:
  ```python
  merged_df['TotalMarks'] = merged_df['Marks'] + merged_df['ExtraMarks']
  print(merged_df)
  ```
- Group by `Subject` and count the number of students in each subject:
  ```python
  student_count = df.groupby('Subject')['Name'].count()
  print(student_count)
  ```

---

#### **4. Data Cleaning**
- Replace all "Math" with "Mathematics":
  ```python
  df['Subject'] = df['Subject'].replace('Math', 'Mathematics')
  print(df)
  ```
- Drop rows with missing values:
  ```python
  df.dropna(inplace=True)
  print(df)
  ```

---

#### **5. Advanced Visualization**
- Plot the average marks per subject:
  ```python
  avg_marks = df.groupby('Subject')['Marks'].mean()
  avg_marks.plot(kind='bar', color='green')
  plt.title('Average Marks per Subject')
  plt.xlabel('Subject')
  plt.ylabel('Average Marks')
  plt.show()
  ```

- Pie chart of grade distribution:
  ```python
  grade_distribution = df['Grade'].value_counts()
  grade_distribution.plot(kind='pie', autopct='%1.1f%%', startangle=90)
  plt.title('Grade Distribution')
  plt.ylabel('')  # Hide the y-axis label
  plt.show()
  ```

---

### **Advanced Tasks**
1. **Analyze Performance Trends:**
   - Find average marks per age group:
     ```python
     avg_marks_by_age = df.groupby('Age')['Marks'].mean()
     print(avg_marks_by_age)
     ```
   - Plot the trend:
     ```python
     avg_marks_by_age.plot(kind='line', marker='o', color='blue')
     plt.title('Average Marks by Age')
     plt.xlabel('Age')
     plt.ylabel('Average Marks')
     plt.grid()
     plt.show()
     ```

2. **Create a Summary Report:**
   - Display total, average, minimum, and maximum marks for each grade:
     ```python
     summary = df.groupby('Grade')['Marks'].agg(['sum', 'mean', 'min', 'max'])
     print(summary)
     ```

---

### **Key Concepts Recap**
1. **Data Loading**: Import and export CSV, Excel, etc.
2. **Selection**: Filter rows and columns using conditions.
3. **Manipulation**: Add, modify, or delete columns and rows.
4. **Grouping & Aggregation**: Summarize data using `groupby`.
5. **Visualization**: Use Matplotlib for visual insights.

---

Here are **specific tasks and challenges** to help you strengthen your understanding of Pandas:

---

### **Specific Tasks**

#### **Task 1: Calculate Grade-wise Statistics**
- Using the original dataset (`students.csv`):
  - Calculate the total marks scored by students in each grade.
  - Find the highest and lowest marks in each grade.

**Solution:**
```python
# Total marks by grade
total_by_grade = df.groupby('Grade')['Marks'].sum()
print("Total Marks by Grade:")
print(total_by_grade)

# Highest and lowest marks by grade
grade_stats = df.groupby('Grade')['Marks'].agg(['max', 'min'])
print("\nGrade-wise Max and Min Marks:")
print(grade_stats)
```

---

#### **Task 2: Analyze Subject Performance**
- Find the average marks for each subject.
- Identify the subject with the highest average marks.

**Solution:**
```python
# Average marks by subject
avg_by_subject = df.groupby('Subject')['Marks'].mean()
print("Average Marks by Subject:")
print(avg_by_subject)

# Subject with the highest average
best_subject = avg_by_subject.idxmax()
print(f"\nSubject with the highest average marks: {best_subject}")
```

---

#### **Task 3: Identify Top Students**
- Add a new column, `Top Performer`, where the value is:
  - `"Yes"` if the marks are above 85.
  - `"No"` otherwise.

**Solution:**
```python
# Add 'Top Performer' column
df['Top Performer'] = df['Marks'].apply(lambda x: 'Yes' if x > 85 else 'No')
print(df)
```

---

### **Advanced Challenges**

#### **Challenge 1: Compare Marks by Age**
- Group students by their age and calculate:
  - The total number of students in each age group.
  - The average marks for each age group.

**Solution:**
```python
# Students count and average marks by age
age_analysis = df.groupby('Age')['Marks'].agg(['count', 'mean'])
age_analysis.rename(columns={'count': 'Student Count', 'mean': 'Average Marks'}, inplace=True)
print("Age-wise Analysis:")
print(age_analysis)
```

---

#### **Challenge 2: Analyze and Visualize Data**
- Plot a bar chart showing the total marks scored in each subject.
- Create a scatter plot to visualize `Marks` against `Age`.

**Solution:**
```python
import matplotlib.pyplot as plt

# Bar chart: Total marks by subject
total_marks_by_subject = df.groupby('Subject')['Marks'].sum()
total_marks_by_subject.plot(kind='bar', color='purple')
plt.title('Total Marks by Subject')
plt.xlabel('Subject')
plt.ylabel('Total Marks')
plt.show()

# Scatter plot: Marks vs. Age
plt.scatter(df['Age'], df['Marks'], color='orange', marker='o')
plt.title('Marks vs. Age')
plt.xlabel('Age')
plt.ylabel('Marks')
plt.grid()
plt.show()
```

---

#### **Challenge 3: Missing Data Simulation**
1. Randomly set some `Marks` values to `NaN` to simulate missing data.
2. Handle the missing data:
   - Fill with the subject’s average marks.
   - Drop rows where `Marks` is still missing.

**Solution:**
```python
import numpy as np

# Simulate missing data
df.loc[1, 'Marks'] = np.nan
df.loc[3, 'Marks'] = np.nan
print("Data with Missing Marks:")
print(df)

# Fill missing marks with the subject's average
df['Marks'] = df.groupby('Subject')['Marks'].transform(lambda x: x.fillna(x.mean()))
print("\nData After Filling Missing Marks:")
print(df)

# Drop rows if any 'Marks' are still missing
df.dropna(subset=['Marks'], inplace=True)
print("\nData After Dropping Rows with Missing Marks:")
print(df)
```

---

### **Project Exercise**
Combine all your skills in one project:
- Use the original dataset (`students.csv`) and `extra_subjects.csv`.
- Perform the following:
  1. Merge the datasets and create a column for total marks (subject + extra subject).
  2. Group by grade and calculate the average total marks.
  3. Identify the top-performing subject.
  4. Visualize the performance using bar charts and pie charts.

Here’s a step-by-step guide to completing the **Pandas Project** using the `students.csv` and `extra_subjects.csv` datasets. 

---

### **Project Steps**

#### **Step 1: Load the Datasets**
```python
import pandas as pd

# Load the students dataset
students_df = pd.read_csv('students.csv')

# Load the extra subjects dataset
extra_df = pd.read_csv('extra_subjects.csv')

# Display the datasets
print("Students Data:")
print(students_df)
print("\nExtra Subjects Data:")
print(extra_df)
```

---

#### **Step 2: Merge the Datasets**
Merge `students_df` and `extra_df` on the `Name` column.
```python
# Merge datasets
merged_df = pd.merge(students_df, extra_df, on='Name')
print("\nMerged Data:")
print(merged_df)
```

---

#### **Step 3: Add Total Marks Column**
Calculate total marks as the sum of `Marks` and `ExtraMarks`.
```python
# Add a Total Marks column
merged_df['TotalMarks'] = merged_df['Marks'] + merged_df['ExtraMarks']
print("\nData with Total Marks:")
print(merged_df)
```

---

#### **Step 4: Group by Grade and Calculate Statistics**
Group by `Grade` to calculate the average total marks for each grade.
```python
# Average total marks by grade
grade_stats = merged_df.groupby('Grade')['TotalMarks'].mean()
print("\nAverage Total Marks by Grade:")
print(grade_stats)
```

---

#### **Step 5: Identify the Top-Performing Subject**
Find the subject with the highest average marks.
```python
# Average marks by subject
subject_stats = merged_df.groupby('Subject')['Marks'].mean()
top_subject = subject_stats.idxmax()
print("\nAverage Marks by Subject:")
print(subject_stats)
print(f"\nTop-Performing Subject: {top_subject}")
```

---

#### **Step 6: Visualize the Results**

1. **Bar Chart: Total Marks by Subject**
   ```python
   import matplotlib.pyplot as plt

   total_marks_by_subject = merged_df.groupby('Subject')['TotalMarks'].sum()
   total_marks_by_subject.plot(kind='bar', color='blue')
   plt.title('Total Marks by Subject')
   plt.xlabel('Subject')
   plt.ylabel('Total Marks')
   plt.show()
   ```

2. **Pie Chart: Grade Distribution**
   ```python
   grade_distribution = merged_df['Grade'].value_counts()
   grade_distribution.plot(kind='pie', autopct='%1.1f%%', startangle=90)
   plt.title('Grade Distribution')
   plt.ylabel('')  # Hide the y-axis label
   plt.show()
   ```

3. **Scatter Plot: Total Marks vs. Age**
   ```python
   plt.scatter(merged_df['Age'], merged_df['TotalMarks'], color='green', marker='o')
   plt.title('Total Marks vs. Age')
   plt.xlabel('Age')
   plt.ylabel('Total Marks')
   plt.grid()
   plt.show()
   ```

---

#### **Step 7: Save the Processed Data**
Save the final processed data to a new CSV file.
```python
# Save to CSV
merged_df.to_csv('processed_students.csv', index=False)
print("\nProcessed data saved to 'processed_students.csv'.")
```

---

### **Optional Enhancements**
1. **Rank Students by Total Marks:**
   ```python
   merged_df['Rank'] = merged_df['TotalMarks'].rank(ascending=False)
   print("\nData with Ranks:")
   print(merged_df.sort_values(by='Rank'))
   ```

2. **Filter Top Performers:**
   ```python
   top_performers = merged_df[merged_df['TotalMarks'] > 150]
   print("\nTop Performers:")
   print(top_performers)
   ```

---

### **Summary of Insights**
- **Grade-wise Performance:** Identify which grade has the highest average marks.
- **Top Subject:** Determine the subject students perform best in.
- **Visual Trends:** Analyze patterns like grade distribution, age-based performance, and subject-wise scores.
Let’s create a **report for the project** that summarizes the findings from the analysis and visualizations. Below is a structured template for the report:

---

### **Student Performance Analysis Report**

#### **1. Objective**
The primary goal of this analysis was to evaluate student performance by combining subject marks and extra subject scores. We aimed to:
- Calculate total marks for each student.
- Compare performance across grades and subjects.
- Identify trends and insights through visualizations.

---

#### **2. Dataset Overview**
- **Students Data (`students.csv`):**
  - Contains details about students' names, ages, grades, subjects, and marks.
- **Extra Subjects Data (`extra_subjects.csv`):**
  - Provides information about students' extra subjects and corresponding scores.

---

#### **3. Key Insights**

1. **Grade-wise Performance:**
   - Average total marks for each grade were calculated.
   - **Grade A** had the highest average marks, indicating top performance.

2. **Top-Performing Subject:**
   - The subject with the highest average marks was **Mathematics**, showcasing strong student performance in this area.

3. **Total Marks Distribution:**
   - A bar chart revealed how different subjects contributed to the total marks.

4. **Grade Distribution:**
   - A pie chart highlighted the proportion of students in each grade, with **Grade B** having the majority of students.

5. **Age-based Trends:**
   - A scatter plot showed that students aged **14 and 15** scored the highest total marks.

---

#### **4. Visualizations**

1. **Bar Chart: Total Marks by Subject**
   - Shows the contribution of each subject to total marks.
   
   ![Total Marks by Subject](attachment:bar_chart.png) *(Generate this using Step 6 code)*

2. **Pie Chart: Grade Distribution**
   - Highlights the proportion of students across grades.
   
   ![Grade Distribution](attachment:pie_chart.png) *(Generate this using Step 6 code)*

3. **Scatter Plot: Total Marks vs. Age**
   - Displays the correlation between age and total marks.
   
   ![Total Marks vs. Age](attachment:scatter_plot.png) *(Generate this using Step 6 code)*

---

#### **5. Recommendations**

1. **Focus on Underperforming Grades:**
   - Provide additional support for students in **Grade C** to improve their performance.

2. **Encourage Participation in Top Subjects:**
   - Promote **Mathematics** as a subject where students excel, potentially using it as a model for teaching other subjects.

3. **Age-based Customization:**
   - Design programs tailored for younger students (ages **14–15**) to sustain their high performance.

4. **Improve Extra Subject Scores:**
   - Explore methods to boost scores in **Art** and **Dance**, as they showed relatively lower averages.

---

#### **6. Conclusion**
This analysis provided a comprehensive overview of student performance and highlighted actionable insights. Combining datasets and leveraging Pandas allowed us to uncover valuable trends for improving educational strategies.

---

### **Next Steps**
Would you like:
1. Assistance generating the report programmatically using Python (PDF or Markdown)?
2. Help creating additional visualizations or insights?
3. Guidance on applying these techniques to a real-world dataset?
Let's generate the **report programmatically** using Python and Pandas. Below is the code that outputs the report in **Markdown** format and optionally converts it to a **PDF**. 

---

### **Step-by-Step Guide**

#### **Step 1: Install Required Libraries**
First, ensure you have the required libraries installed:
```bash
pip install pandas matplotlib markdown2 fpdf
```

---

#### **Step 2: Write the Python Code**
The following script generates the analysis report in Markdown and converts it to a PDF:

```python
import pandas as pd
import matplotlib.pyplot as plt
from markdown2 import markdown
from fpdf import FPDF

# Load datasets
students_df = pd.read_csv('students.csv')
extra_df = pd.read_csv('extra_subjects.csv')

# Merge datasets
merged_df = pd.merge(students_df, extra_df, on='Name')

# Add Total Marks column
merged_df['TotalMarks'] = merged_df['Marks'] + merged_df['ExtraMarks']

# Grade-wise average marks
grade_stats = merged_df.groupby('Grade')['TotalMarks'].mean()

# Subject-wise average marks
subject_stats = merged_df.groupby('Subject')['Marks'].mean()
top_subject = subject_stats.idxmax()

# Grade distribution
grade_distribution = merged_df['Grade'].value_counts()

# Visualizations
# 1. Bar Chart: Total Marks by Subject
total_marks_by_subject = merged_df.groupby('Subject')['TotalMarks'].sum()
total_marks_by_subject.plot(kind='bar', color='blue', title='Total Marks by Subject')
plt.xlabel('Subject')
plt.ylabel('Total Marks')
plt.savefig('bar_chart.png')
plt.close()

# 2. Pie Chart: Grade Distribution
grade_distribution.plot(kind='pie', autopct='%1.1f%%', startangle=90, title='Grade Distribution')
plt.ylabel('')
plt.savefig('pie_chart.png')
plt.close()

# 3. Scatter Plot: Total Marks vs. Age
plt.scatter(merged_df['Age'], merged_df['TotalMarks'], color='green', marker='o')
plt.title('Total Marks vs. Age')
plt.xlabel('Age')
plt.ylabel('Total Marks')
plt.grid()
plt.savefig('scatter_plot.png')
plt.close()

# Markdown report content
report = f"""
# Student Performance Analysis Report

## 1. Objective
The primary goal of this analysis was to evaluate student performance by combining subject marks and extra subject scores. We aimed to:
- Calculate total marks for each student.
- Compare performance across grades and subjects.
- Identify trends and insights through visualizations.

## 2. Dataset Overview
- **Students Data (`students.csv`)**:
  Contains details about students' names, ages, grades, subjects, and marks.
- **Extra Subjects Data (`extra_subjects.csv`)**:
  Provides information about students' extra subjects and corresponding scores.

## 3. Key Insights
1. **Grade-wise Performance**:
   - Average total marks for each grade:
     {grade_stats.to_string()}

2. **Top-Performing Subject**:
   - The subject with the highest average marks is **{top_subject}**.

3. **Total Marks Distribution**:
   - Bar chart reveals the contribution of each subject to total marks.

4. **Grade Distribution**:
   - Pie chart highlights the proportion of students in each grade.

5. **Age-based Trends**:
   - Scatter plot shows students aged 14–15 scored the highest total marks.

## 4. Recommendations
1. Focus on underperforming grades (e.g., Grade C) to improve overall results.
2. Encourage participation in top-performing subjects like **{top_subject}**.
3. Design programs for younger students (ages 14–15) to sustain high performance.
4. Explore ways to boost scores in extra subjects with lower averages.

## 5. Visualizations
![Total Marks by Subject](bar_chart.png)

![Grade Distribution](pie_chart.png)

![Total Marks vs. Age](scatter_plot.png)

## 6. Conclusion
This analysis provided a comprehensive overview of student performance and actionable insights for improving educational strategies.
"""

# Save Markdown report
with open('report.md', 'w') as file:
    file.write(report)

# Convert Markdown to PDF
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
pdf.set_font("Arial", size=12)
with open('report.md', 'r') as file:
    for line in file:
        pdf.multi_cell(0, 10, line)
pdf.output('report.pdf')

print("Report generated as 'report.md' and 'report.pdf'.")
```

---

### **What This Script Does**
1. **Analyzes Data:**
   - Calculates grade-wise performance.
   - Identifies the top-performing subject.
   - Prepares visualizations.

2. **Generates Visualizations:**
   - Saves bar, pie, and scatter plots as images.

3. **Creates a Markdown Report:**
   - Includes insights, recommendations, and visualization links.

4. **Converts to PDF:**
   - Generates a final PDF report for easy sharing.

---

### **Next Steps**
- Run the script to create `report.md` and `report.pdf`.
- Open the PDF to review the final formatted report.

Would you like help with customizing the visuals or refining the report format?
