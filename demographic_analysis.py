import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load the new customers data
customers = pd.read_csv('Customers.csv', encoding='ISO-8859-1')

# Calculate Age from Birthday
customers['Birthday'] = pd.to_datetime(customers['Birthday'], errors='coerce')
today = pd.to_datetime('today')
customers['Age'] = customers['Birthday'].apply(lambda x: today.year - x.year if pd.notnull(x) else None)

# Drop rows with missing age or gender
customers = customers.dropna(subset=['Age', 'Gender'])

# Gender distribution
gender_counts = customers['Gender'].value_counts()
plt.figure(figsize=(6, 4))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
plt.title('Gender Distribution')
plt.show()

# Age distribution
plt.figure(figsize=(8, 4))
sns.histplot(customers['Age'], bins=20, kde=True, color='skyblue')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Age group distribution
bins = [0, 18, 25, 35, 45, 55, 65, 100]
labels = ['<18', '18-25', '26-35', '36-45', '46-55', '56-65', '65+']
customers['AgeGroup'] = pd.cut(customers['Age'], bins=bins, labels=labels, right=False)
age_group_counts = customers['AgeGroup'].value_counts().sort_index()
plt.figure(figsize=(8, 4))
sns.barplot(x=age_group_counts.index, y=age_group_counts.values, palette='viridis')
plt.title('Age Group Distribution')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.show()

# Top 10 cities
top_cities = customers['City'].value_counts().head(10)
plt.figure(figsize=(10, 5))
sns.barplot(x=top_cities.values, y=top_cities.index, palette='mako')
plt.title('Top 10 Cities by Customer Count')
plt.xlabel('Number of Customers')
plt.ylabel('City')
plt.show()

# Top 10 states
top_states = customers['State'].value_counts().head(10)
plt.figure(figsize=(10, 5))
sns.barplot(x=top_states.values, y=top_states.index, palette='crest')
plt.title('Top 10 States by Customer Count')
plt.xlabel('Number of Customers')
plt.ylabel('State')
plt.show()

# Country distribution
country_counts = customers['Country'].value_counts().head(10)
plt.figure(figsize=(10, 5))
sns.barplot(x=country_counts.values, y=country_counts.index, palette='flare')
plt.title('Top 10 Countries by Customer Count')
plt.xlabel('Number of Customers')
plt.ylabel('Country')
plt.show()

# Continent distribution
continent_counts = customers['Continent'].value_counts()
plt.figure(figsize=(8, 4))
sns.barplot(x=continent_counts.index, y=continent_counts.values, palette='Set2')
plt.title('Continent Distribution')
plt.xlabel('Continent')
plt.ylabel('Number of Customers')
plt.show()

# Print summary statistics
total_customers = len(customers)
print(f"Total customers: {total_customers}")
print("\nGender distribution:")
print(gender_counts)
print("\nAge group distribution:")
print(age_group_counts)
print("\nTop 10 cities:")
print(top_cities)
print("\nTop 10 states:")
print(top_states)
print("\nTop 10 countries:")
print(country_counts)
print("\nContinent distribution:")
print(continent_counts) 