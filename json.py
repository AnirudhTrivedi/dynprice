import pandas as pd
import json

# Load the dataset
df = pd.read_csv("airline_booking_dataset.csv")

# Generate statistics
stats = {
    "Basic_Statistics": df.describe().to_dict(),
    "Traveler_Type_Counts": df["Traveler_Type"].value_counts().to_dict(),
    "Membership_Status_Counts": df["Membership_Status"].value_counts().to_dict(),
    "Membership_Status_by_Traveler_Type": df.groupby("Traveler_Type")["Membership_Status"].value_counts().to_dict(),
    "Travel_Class_Counts": df["Travel_Class"].value_counts().to_dict(),
    "Peak_Season_Bookings": df["Peak_Season"].value_counts().to_dict(),
    "Average_Ticket_Price": df["Final_Ticket_Price"].mean(),
    "Ancillary_Service_Purchases": {
        "WiFi": df["WiFi_Purchased"].value_counts().to_dict(),
        "Lounge_Access": df["Lounge_Access_Purchased"].value_counts().to_dict(),
        "Meals": df["Meals_Purchased"].value_counts().to_dict(),
        "Insurance": df["Insurance_Purchased"].value_counts().to_dict(),
        "Priority_Boarding": df["Priority_Boarding_Purchased"].value_counts().to_dict(),
        "Carbon_Offset": df["Carbon_Offset_Purchased"].value_counts().to_dict()
    },
    "Booking_Behavior": {
        "Average_Days_Before_Travel": df["Days_Before_Travel"].mean(),
        "Booking_within_48hrs": df["Booking_within_48hrs"].value_counts().to_dict()
    },
    "Ancillary_Spending": {
        "Total_Ancillary_Cost_Average": df["Total_Ancillary_Cost"].mean(),
        "Total_Ancillary_Cost_Sum": df["Total_Ancillary_Cost"].sum()
    }
}

# Save statistics to JSON file
with open("airline_booking_stats.json", "w") as json_file:
    json.dump(stats, json_file, indent=4)

print("\nData analysis completed! JSON statistics saved as 'airline_booking_stats.json'.")
