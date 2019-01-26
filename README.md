# Salary-by-resume
Machine learning: salary forecasting by the description of skills in the resume (without regard to work experience)

Condition: according to the description of skills in the resume, predict the paylock of the Python developer. Resume 
programmatically download from any Ukrainian job search site.

For the solution, the libraries BeautifulSoup, urllib.request, sklearn.linear_model, numpy, pandas were used.
The solution is based on the concept of linear regression. For training the model used resumes with salaries indicated.

skills.csv - a file with a list of skills and points awarded for them.
