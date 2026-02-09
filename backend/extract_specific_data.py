import pandas as pd
import os

# Define paths
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "../data")
input_file = os.path.join(data_dir, "cleaned_medicine_data.csv")
output_file = os.path.join(data_dir, "specific_medicine_data.csv")

# List of medicines provided in your instruction
target_medicines = [
    # List 1: Major Diseases
    "Metformin", "Glimepiride", "Sitagliptin",
    "Amlodipine", "Losartan", "Atenolol",
    "Aspirin", "Clopidogrel", "Atorvastatin",
    "Isoniazid", "Rifampicin", "Ethambutol",
    "Salbutamol Inhaler", "Budesonide", "Formoterol",
    "Cisplatin", "Carboplatin", "Paclitaxel",
    "Paracetamol", "Acetaminophen",
    "Chloroquine", "Artemether-Lumefantrine",
    "Levothyroxine", "Liothyronine",
    "Sertraline", "Fluoxetine", "Escitalopram",
    
    # List 2: Common Diseases
    "Cetirizine", "Loratadine",
    "Sumatriptan", "Rizatriptan",
    "ORS", "Zinc Tablets",
    "Omeprazole", "Pantoprazole",
    "Nitrofurantoin", "Ciprofloxacin",
    "Levocetirizine",
    "Loperamide",
    "Ibuprofen", "Diclofenac",
    "Ciprofloxacin Eye Drops", "Ofloxacin Eye Drops"
]

# Mapping of medicines to their substitutes based on your list
substitutes_mapping = {
    "Metformin": "Glimepiride, Sitagliptin",
    "Amlodipine": "Losartan, Atenolol",
    "Aspirin": "Clopidogrel, Atorvastatin",
    "Isoniazid": "Rifampicin, Ethambutol",
    "Salbutamol Inhaler": "Budesonide, Formoterol",
    "Cisplatin": "Carboplatin, Paclitaxel",
    "Paracetamol": "Acetaminophen",
    "Chloroquine": "Artemether-Lumefantrine",
    "Levothyroxine": "Liothyronine",
    "Sertraline": "Fluoxetine, Escitalopram",
    "Cetirizine": "Loratadine, Levocetirizine",
    "Sumatriptan": "Rizatriptan",
    "ORS": "Zinc Tablets, Loperamide",
    "Omeprazole": "Pantoprazole",
    "Nitrofurantoin": "Ciprofloxacin",
    "Ibuprofen": "Diclofenac",
    "Ciprofloxacin Eye Drops": "Ofloxacin Eye Drops"
}

def extract_and_save():
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"❌ Error: Input file '{input_file}' not found.")
        print("   Please run 'clean_data.py' first or ensure the file exists.")
        return

    print(f"📖 Reading '{input_file}'...")
    try:
        df = pd.read_csv(input_file, low_memory=False)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    # Normalize column names in the CSV
    df.columns = df.columns.str.strip()
    
    # Identify the Medicine Name column
    med_col = None
    for col in ['Medicine Name', 'Drug', 'Drug Name', 'drugName', 'Medicine']:
        if col in df.columns:
            med_col = col
            break
    
    if not med_col:
        print("❌ Error: Could not find a 'Medicine Name' column in the source CSV.")
        return

    print(f"🔍 Searching for {len(target_medicines)} specific medicines...")

    # Create a lowercase map for case-insensitive matching
    target_meds_lower = {m.lower(): m for m in target_medicines}
    
    # Filter the dataframe
    # We use a lambda to check if the medicine name (lowercased) is in our target list
    mask = df[med_col].astype(str).str.lower().isin(target_meds_lower.keys())
    filtered_df = df[mask].copy()

    # Check which medicines were found
    found_meds = filtered_df[med_col].astype(str).str.lower().unique()
    found_names = [target_meds_lower[m] for m in found_meds if m in target_meds_lower]
    
    print(f"✅ Found {len(found_names)} unique medicines from your list in the dataset.")
    
    missing = set(target_medicines) - set(found_names)
    if missing:
        print(f"⚠️ The following medicines were NOT found in the source data:\n   {', '.join(missing)}")

    if filtered_df.empty:
        print("❌ No matching records found. Exiting.")
        return

    # Update Substitute column with the mapping
    print("🔄 Updating Substitute information (verifying availability in source data)...")
    if 'Substitute' not in filtered_df.columns:
        filtered_df['Substitute'] = None

    # Get all available medicines from the FULL source dataframe
    all_source_meds = set(df[med_col].astype(str).str.lower().unique())

    for med, subs in substitutes_mapping.items():
        # Split provided substitutes
        potential_subs = [s.strip() for s in subs.split(',')]
        
        # Filter: Keep only substitutes that exist in the source dataset
        confirmed_subs = [s for s in potential_subs if s.lower() in all_source_meds]
        
        if confirmed_subs:
            valid_subs_str = ", ".join(confirmed_subs)
            # Case-insensitive match for medicine name
            mask = filtered_df[med_col].astype(str).str.lower() == med.lower()
            if mask.any():
                filtered_df.loc[mask, 'Substitute'] = valid_subs_str

    # Ensure we have exactly 1500 rows
    target_rows = 1500
    print(f"📊 Total matching rows found in source: {len(filtered_df)}")

    # Sample 1500 rows (with replacement if we have fewer than 1500, without if we have more)
    replace_flag = len(filtered_df) < target_rows
    final_df = filtered_df.sample(n=target_rows, replace=replace_flag, random_state=42)

    # Save to new CSV
    print(f"💾 Saving extracted data to '{output_file}'...")
    final_df.to_csv(output_file, index=False)
    print(f"✅ Done! Created '{os.path.basename(output_file)}' with {len(final_df)} rows.")

if __name__ == "__main__":
    extract_and_save()