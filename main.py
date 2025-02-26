import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("airline_booking_dataset.csv")

# Summary Statistics
print("\n### Basic Statistics ###")
print(df.describe())

# Count of traveler types
print("\n### Traveler Type Counts ###")
print(df["Traveler_Type"].value_counts())

# Count of membership statuses
print("\n### Membership Status Counts ###")
print(df["Membership_Status"].value_counts())

# Count of travel classes
print("\n### Travel Class Counts ###")
print(df["Travel_Class"].value_counts())

# Peak season vs non-peak season bookings
print("\n### Peak Season Bookings ###")
print(df["Peak_Season"].value_counts())

# Average ticket price
print("\n### Average Ticket Price ###")
print(df["Final_Ticket_Price"].mean())

# Distribution of ticket prices
plt.figure(figsize=(10,5))
sns.histplot(df["Final_Ticket_Price"], bins=50, kde=True, color='blue')
plt.title("Distribution of Final Ticket Prices")
plt.xlabel("Ticket Price ($)")
plt.ylabel("Frequency")
plt.show()

# Boxplot for ticket prices by traveler type
plt.figure(figsize=(10,5))
sns.boxplot(x=df["Traveler_Type"], y=df["Final_Ticket_Price"], palette='Set2')
plt.title("Ticket Prices by Traveler Type")
plt.show()

# Booking behavior: Days before travel
plt.figure(figsize=(10,5))
sns.histplot(df["Days_Before_Travel"], bins=50, kde=True, color='green')
plt.title("Booking Behavior: Days Before Travel")
plt.xlabel("Days Before Travel")
plt.ylabel("Frequency")
plt.show()

# Ancillary service purchase patterns
ancillary_features = ["WiFi_Purchased", "Lounge_Access_Purchased", "Meals_Purchased", "Insurance_Purchased", "Priority_Boarding_Purchased", "Carbon_Offset_Purchased"]
for feature in ancillary_features:
    plt.figure(figsize=(6,4))
    sns.countplot(x=df[feature], palette='coolwarm')
    plt.title(f"Distribution of {feature.replace('_', ' ')}")
    plt.show()

# Seasonal effect on bookings
plt.figure(figsize=(10,5))
sns.countplot(x=df["Peak_Season"], hue=df["Traveler_Type"], palette='muted')
plt.title("Peak Season vs. Non-Peak Season Bookings")
plt.show()

# Analysis of booking lead time by travel class
plt.figure(figsize=(10,5))
sns.boxplot(x=df["Travel_Class"], y=df["Days_Before_Travel"], palette='coolwarm')
plt.title("Booking Lead Time by Travel Class")
plt.xlabel("Travel Class")
plt.ylabel("Days Before Travel")
plt.show()

# Correlation heatmap of numerical features
plt.figure(figsize=(12,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Features")
plt.show()

# Analysis of total ancillary spending
plt.figure(figsize=(10,5))
sns.histplot(df["Total_Ancillary_Cost"], bins=50, kde=True, color='purple')
plt.title("Distribution of Total Ancillary Spending")
plt.xlabel("Total Ancillary Cost ($)")
plt.ylabel("Frequency")
plt.show()

print("\nData analysis completed! Charts and insights generated.")
