#!/usr/bin/env python3
"""
create_db.py — Build the DrugBank lite SQLite database used by PharmaAgentEnvironment.

This script creates drugbank_lite.db with two tables:
  • drugs         — drug name, indication, status, type
  • interactions  — drug1_name, drug2_name, description

Run once before starting the server:
    python create_db.py

If you already have a drugbank_lite.db (from seeding the full DrugBank data),
skip this and just set DB_PATH to point to it.
"""
import os
import sqlite3

DB_PATH = os.environ.get("DB_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "drugbank_lite.db"))

# ── Drug seed data ─────────────────────────────────────────────────────────────
DRUGS = [
    # Hypertension
    ("Lisinopril",     "Treatment of hypertension and heart failure. Reduces blood pressure in patients with high blood pressure.", "approved", "small molecule"),
    ("Amlodipine",     "Treatment of hypertension and angina. Calcium channel blocker used for high blood pressure.", "approved", "small molecule"),
    ("Losartan",       "Treatment of hypertension. ARB used for high blood pressure, diabetic nephropathy.", "approved", "small molecule"),
    ("Metoprolol",     "Treatment of hypertension, angina pectoris, heart failure. Beta-blocker for blood pressure.", "approved", "small molecule"),
    ("Hydrochlorothiazide", "Treatment of hypertension and edema. Diuretic for high blood pressure.", "approved", "small molecule"),
    ("Atenolol",       "Treatment of hypertension and angina. Beta-1 selective blocker.", "approved", "small molecule"),
    ("Ramipril",       "Treatment of hypertension and heart failure. ACE inhibitor.", "approved", "small molecule"),
    ("Valsartan",      "Treatment of hypertension and heart failure. Angiotensin II receptor antagonist.", "approved", "small molecule"),
    ("Diltiazem",      "Treatment of hypertension and angina. Calcium channel blocker.", "approved", "small molecule"),
    ("Carvedilol",     "Treatment of hypertension and chronic heart failure. Alpha and beta blocker.", "approved", "small molecule"),

    # Diabetes
    ("Metformin",      "First-line treatment for type 2 diabetes mellitus. Reduces hepatic glucose production and improves hyperglycemia.", "approved", "small molecule"),
    ("Glipizide",      "Treatment of type 2 diabetes mellitus. Sulfonylurea that stimulates insulin secretion.", "approved", "small molecule"),
    ("Sitagliptin",    "Treatment of type 2 diabetes mellitus. DPP-4 inhibitor for glycemic control.", "approved", "small molecule"),
    ("Empagliflozin",  "Treatment of type 2 diabetes mellitus. SGLT2 inhibitor for hyperglycemia.", "approved", "small molecule"),
    ("Glibenclamide",  "Treatment of type 2 diabetes mellitus. Sulfonylurea for glycemic control.", "approved", "small molecule"),
    ("Pioglitazone",   "Treatment of type 2 diabetes mellitus. Thiazolidinedione for insulin resistance.", "approved", "small molecule"),
    ("Dapagliflozin",  "Treatment of type 2 diabetes mellitus and heart failure. SGLT2 inhibitor.", "approved", "small molecule"),
    ("Liraglutide",    "Treatment of type 2 diabetes mellitus. GLP-1 receptor agonist.", "approved", "small molecule"),

    # Heart Failure
    ("Furosemide",     "Treatment of heart failure, edema, and hypertension. Loop diuretic.", "approved", "small molecule"),
    ("Spironolactone", "Treatment of heart failure and edema. Aldosterone antagonist.", "approved", "small molecule"),
    ("Digoxin",        "Treatment of heart failure and atrial fibrillation. Cardiac glycoside.", "approved", "small molecule"),
    ("Eplerenone",     "Treatment of heart failure and hypertension. Selective aldosterone blocker.", "approved", "small molecule"),
    ("Sacubitril",     "Treatment of chronic heart failure with reduced ejection fraction.", "approved", "small molecule"),
    ("Ivabradine",     "Treatment of chronic heart failure. Reduces heart rate.", "approved", "small molecule"),

    # Rheumatoid Arthritis
    ("Methotrexate",   "Treatment of rheumatoid arthritis and certain cancers. Disease-modifying antirheumatic drug.", "approved", "small molecule"),
    ("Hydroxychloroquine", "Treatment of rheumatoid arthritis and lupus. Antimalarial and DMARD.", "approved", "small molecule"),
    ("Sulfasalazine",  "Treatment of rheumatoid arthritis and inflammatory bowel disease. DMARD.", "approved", "small molecule"),
    ("Leflunomide",    "Treatment of rheumatoid arthritis. Pyrimidine synthesis inhibitor, DMARD.", "approved", "small molecule"),
    ("Naproxen",       "Treatment of pain and inflammation including rheumatoid arthritis. NSAID.", "approved", "small molecule"),
    ("Ibuprofen",      "Treatment of pain, fever, and inflammation. NSAID used in arthritis.", "approved", "small molecule"),
    ("Celecoxib",      "Treatment of rheumatoid arthritis and osteoarthritis. COX-2 inhibitor NSAID.", "approved", "small molecule"),

    # Asthma
    ("Salbutamol",     "Treatment of bronchospasm in asthma and COPD. Short-acting beta-2 agonist.", "approved", "small molecule"),
    ("Salmeterol",     "Maintenance treatment of bronchial asthma. Long-acting beta-2 agonist.", "approved", "small molecule"),
    ("Budesonide",     "Treatment of bronchial asthma. Inhaled corticosteroid for airway inflammation.", "approved", "small molecule"),
    ("Fluticasone",    "Treatment of asthma and allergic rhinitis. Inhaled corticosteroid.", "approved", "small molecule"),
    ("Montelukast",    "Prophylaxis and treatment of asthma and allergic rhinitis. Leukotriene receptor antagonist.", "approved", "small molecule"),
    ("Ipratropium",    "Treatment of bronchospasm in COPD and asthma. Anticholinergic bronchodilator.", "approved", "small molecule"),
    ("Theophylline",   "Treatment of asthma and COPD. Bronchodilator methylxanthine.", "approved", "small molecule"),

    # Epilepsy
    ("Carbamazepine",  "Treatment of epilepsy and trigeminal neuralgia. Anticonvulsant mood stabiliser.", "approved", "small molecule"),
    ("Valproic Acid",  "Treatment of epilepsy, bipolar disorder. Anticonvulsant and mood stabiliser.", "approved", "small molecule"),
    ("Lamotrigine",    "Treatment of epilepsy and bipolar disorder. Antiepileptic agent.", "approved", "small molecule"),
    ("Phenytoin",      "Treatment of epilepsy and seizures. Anticonvulsant sodium channel blocker.", "approved", "small molecule"),
    ("Levetiracetam",  "Treatment of epilepsy. Antiepileptic with broad-spectrum activity.", "approved", "small molecule"),
    ("Topiramate",     "Treatment of epilepsy and migraine prevention. Anticonvulsant.", "approved", "small molecule"),
    ("Clonazepam",     "Treatment of epilepsy and panic disorder. Benzodiazepine anticonvulsant.", "approved", "small molecule"),

    # Hypothyroidism
    ("Levothyroxine",  "Replacement therapy for hypothyroidism. Synthetic thyroid hormone (T4).", "approved", "small molecule"),
    ("Liothyronine",   "Treatment of hypothyroidism. Synthetic thyroid hormone (T3).", "approved", "small molecule"),

    # Depression
    ("Sertraline",     "Treatment of major depressive disorder, OCD, PTSD. SSRI antidepressant.", "approved", "small molecule"),
    ("Fluoxetine",     "Treatment of major depressive disorder and OCD. SSRI antidepressant.", "approved", "small molecule"),
    ("Escitalopram",   "Treatment of major depressive disorder and anxiety. SSRI antidepressant.", "approved", "small molecule"),
    ("Venlafaxine",    "Treatment of major depressive disorder and anxiety. SNRI antidepressant.", "approved", "small molecule"),
    ("Bupropion",      "Treatment of major depressive disorder and smoking cessation. NDRI antidepressant.", "approved", "small molecule"),
    ("Mirtazapine",    "Treatment of major depressive disorder. NaSSA antidepressant.", "approved", "small molecule"),
    ("Duloxetine",     "Treatment of depression, anxiety and neuropathic pain. SNRI antidepressant.", "approved", "small molecule"),
    ("Amitriptyline",  "Treatment of depression and chronic pain. Tricyclic antidepressant.", "approved", "small molecule"),

    # Peptic Ulcer
    ("Omeprazole",     "Treatment of peptic ulcer disease, GERD, and H. pylori infection. Proton pump inhibitor.", "approved", "small molecule"),
    ("Pantoprazole",   "Treatment of gastric ulcer, GERD. Proton pump inhibitor for acid reduction.", "approved", "small molecule"),
    ("Esomeprazole",   "Treatment of GERD and peptic ulcer. Proton pump inhibitor.", "approved", "small molecule"),
    ("Ranitidine",     "Treatment of peptic ulcer and GERD. H2 receptor antagonist.", "approved", "small molecule"),
    ("Famotidine",     "Treatment of peptic ulcer disease and GERD. H2 blocker.", "approved", "small molecule"),
    ("Clarithromycin", "Treatment of H. pylori infection and respiratory infections. Macrolide antibiotic.", "approved", "small molecule"),
    ("Amoxicillin",    "Treatment of bacterial infections including H. pylori. Broad-spectrum penicillin.", "approved", "small molecule"),

    # Atrial Fibrillation
    ("Warfarin",       "Anticoagulation therapy for atrial fibrillation, DVT, PE prevention.", "approved", "small molecule"),
    ("Rivaroxaban",    "Anticoagulation for atrial fibrillation and DVT. Direct factor Xa inhibitor.", "approved", "small molecule"),
    ("Apixaban",       "Anticoagulation for atrial fibrillation. Factor Xa inhibitor.", "approved", "small molecule"),
    ("Dabigatran",     "Anticoagulation for atrial fibrillation. Direct thrombin inhibitor.", "approved", "small molecule"),
    ("Amiodarone",     "Treatment of atrial fibrillation and ventricular arrhythmia. Antiarrhythmic.", "approved", "small molecule"),
    ("Flecainide",     "Treatment of atrial fibrillation and arrhythmia. Class IC antiarrhythmic.", "approved", "small molecule"),
    ("Bisoprolol",     "Rate control in atrial fibrillation and heart failure. Beta-1 selective blocker.", "approved", "small molecule"),
    ("Dronedarone",    "Maintenance of sinus rhythm in atrial fibrillation. Antiarrhythmic.", "approved", "small molecule"),

    # Common additional drugs
    ("Atorvastatin",   "Treatment of hyperlipidemia and cardiovascular disease prevention. HMG-CoA reductase inhibitor.", "approved", "small molecule"),
    ("Simvastatin",    "Treatment of hyperlipidemia. HMG-CoA reductase inhibitor statin.", "approved", "small molecule"),
    ("Aspirin",        "Antiplatelet therapy for cardiovascular disease prevention. Pain and fever treatment.", "approved", "small molecule"),
    ("Clopidogrel",    "Antiplatelet therapy for ACS and stroke prevention. P2Y12 inhibitor.", "approved", "small molecule"),
    ("Allopurinol",    "Treatment of gout and hyperuricemia. Xanthine oxidase inhibitor.", "approved", "small molecule"),
    ("Gabapentin",     "Treatment of epilepsy and neuropathic pain. Anticonvulsant.", "approved", "small molecule"),
    ("Prednisolone",   "Anti-inflammatory and immunosuppressive therapy. Corticosteroid.", "approved", "small molecule"),
    ("Dexamethasone",  "Anti-inflammatory corticosteroid for various inflammatory conditions.", "approved", "small molecule"),
]

# ── Interaction seed data ──────────────────────────────────────────────────────
INTERACTIONS = [
    # Warfarin (major/contraindicated)
    ("Warfarin", "Amiodarone",
     "Amiodarone markedly increases warfarin plasma levels and anticoagulant effect. This is a major interaction that can cause life-threatening bleeding. Contraindicated or requires extreme caution and dose reduction."),
    ("Warfarin", "Methotrexate",
     "Methotrexate may increase the anticoagulant effect of warfarin leading to serious bleeding risk. Major interaction — monitor INR closely."),
    ("Warfarin", "Clarithromycin",
     "Clarithromycin inhibits CYP3A4 and can significantly increase warfarin levels causing major bleeding risk. Serious interaction."),
    ("Warfarin", "Aspirin",
     "Aspirin combined with warfarin substantially increases bleeding risk. Major interaction — avoid combination unless specifically indicated."),
    ("Warfarin", "Ibuprofen",
     "NSAIDs including ibuprofen increase bleeding risk when combined with warfarin. Major interaction."),
    ("Warfarin", "Naproxen",
     "Naproxen NSAID combined with warfarin significantly increases risk of serious bleeding. Major interaction."),

    # Metformin interactions
    ("Metformin", "Alcohol",
     "Alcohol increases the risk of lactic acidosis when combined with metformin. Moderate interaction."),

    # Digoxin (major)
    ("Digoxin", "Amiodarone",
     "Amiodarone significantly increases digoxin levels leading to digoxin toxicity. Major interaction — reduce digoxin dose by 50% and monitor."),
    ("Digoxin", "Clarithromycin",
     "Clarithromycin inhibits P-glycoprotein and raises digoxin levels substantially. Major toxicity risk."),

    # SSRI interactions
    ("Fluoxetine", "Tramadol",
     "Combination may cause serotonin syndrome, a potentially life-threatening condition. Contraindicated."),
    ("Sertraline", "Tramadol",
     "Sertraline with tramadol increases risk of serotonin syndrome. Major interaction — avoid."),
    ("Amitriptyline", "MAOIs",
     "Combination of tricyclic antidepressants with MAOIs can cause severe serotonin syndrome. Contraindicated."),

    # Anticoagulants
    ("Rivaroxaban", "Aspirin",
     "Aspirin combined with rivaroxaban increases bleeding risk significantly. Use caution, monitor for bleeding."),
    ("Apixaban", "Clopidogrel",
     "Dual antiplatelet plus anticoagulant therapy greatly increases bleeding risk. Major interaction."),
    ("Dabigatran", "Amiodarone",
     "Amiodarone increases dabigatran exposure through P-glycoprotein inhibition. Dose adjustment may be needed. Moderate to major interaction."),

    # Methotrexate interactions
    ("Methotrexate", "Naproxen",
     "Naproxen reduces methotrexate elimination and raises its toxicity risk. Major interaction."),
    ("Methotrexate", "Ibuprofen",
     "Ibuprofen impairs methotrexate renal excretion, increasing toxicity. Major interaction."),
    ("Methotrexate", "Trimethoprim",
     "Trimethoprim inhibits dihydrofolate reductase, additive toxicity with methotrexate. Contraindicated."),

    # Antiepileptics
    ("Carbamazepine", "Valproic Acid",
     "Carbamazepine induces metabolism of valproic acid, reducing its levels. Complex interaction, monitor levels."),
    ("Carbamazepine", "Warfarin",
     "Carbamazepine is a potent CYP inducer, significantly reducing warfarin efficacy. Major interaction."),
    ("Carbamazepine", "Amiodarone",
     "Carbamazepine may reduce amiodarone levels through CYP induction. Monitor effectiveness."),
    ("Phenytoin", "Warfarin",
     "Phenytoin alters warfarin metabolism unpredictably. Initial inhibition then induction. Major interaction, monitor INR."),
    ("Valproic Acid", "Lamotrigine",
     "Valproic acid inhibits lamotrigine metabolism, doubling lamotrigine levels and increasing toxicity risk. Major — reduce lamotrigine dose."),

    # Statins
    ("Simvastatin", "Amiodarone",
     "Amiodarone inhibits CYP3A4, increasing simvastatin exposure and risk of myopathy/rhabdomyolysis. Contraindicated at high simvastatin doses."),
    ("Atorvastatin", "Clarithromycin",
     "Clarithromycin strongly inhibits CYP3A4, raising atorvastatin levels and rhabdomyolysis risk. Major interaction."),

    # ACE inhibitor + potassium-sparing
    ("Lisinopril", "Spironolactone",
     "Combination of ACE inhibitor with aldosterone antagonist significantly increases risk of life-threatening hyperkalaemia. Major interaction — monitor potassium levels closely."),
    ("Ramipril", "Spironolactone",
     "This combination greatly increases hyperkalaemia risk. Major interaction — requires careful potassium monitoring."),

    # Thyroid
    ("Levothyroxine", "Amiodarone",
     "Amiodarone contains large amounts of iodine and inhibits T4 to T3 conversion. Can cause hypo or hyperthyroidism. Major interaction."),
    ("Levothyroxine", "Calcium carbonate",
     "Calcium reduces absorption of levothyroxine. Separate doses by at least 4 hours. Moderate interaction."),

    # PPI + Clopidogrel
    ("Omeprazole", "Clopidogrel",
     "Omeprazole inhibits CYP2C19 and reduces clopidogrel activation, decreasing its antiplatelet effect. Moderate to major interaction — consider alternative PPI."),

    # Antibiotics
    ("Clarithromycin", "Carbamazepine",
     "Clarithromycin inhibits CYP3A4, significantly increasing carbamazepine levels and toxicity. Major interaction."),

    # Theophylline
    ("Theophylline", "Clarithromycin",
     "Clarithromycin inhibits theophylline metabolism, raising theophylline levels and toxicity risk. Major interaction."),
    ("Theophylline", "Ciprofloxacin",
     "Ciprofloxacin raises theophylline plasma levels significantly. Major interaction — reduce theophylline dose."),

    # Beta blocker + Amiodarone
    ("Bisoprolol", "Amiodarone",
     "Amiodarone combined with beta-blockers (bisoprolol) increases risk of bradycardia, AV block, and cardiac arrest. Major interaction."),
    ("Metoprolol", "Amiodarone",
     "Amiodarone inhibits CYP2D6, increasing metoprolol exposure. Combination increases risk of bradycardia. Major interaction."),

    # Dronedarone
    ("Dronedarone", "Simvastatin",
     "Dronedarone inhibits CYP3A4, raising simvastatin levels and risk of myopathy. Major interaction."),
    ("Dronedarone", "Warfarin",
     "Dronedarone increases anticoagulant effect of warfarin. Major interaction — monitor INR."),

    # Flecainide
    ("Flecainide", "Amiodarone",
     "Amiodarone inhibits flecainide metabolism and has additive cardiac effects. Major interaction — avoid combination."),

    # Hypoglycaemia
    ("Metformin", "Glipizide",
     "Combination of metformin with sulfonylurea (glipizide) increases hypoglycaemia risk. Monitor blood glucose. Moderate interaction."),
    ("Glipizide", "Fluconazole",
     "Fluconazole inhibits CYP2C9, significantly increasing glipizide levels and hypoglycaemia risk. Major interaction."),

    # Allopurinol
    ("Allopurinol", "Azathioprine",
     "Allopurinol inhibits xanthine oxidase, drastically increasing azathioprine toxicity. Contraindicated or require 75% azathioprine dose reduction."),
    ("Allopurinol", "Amoxicillin",
     "Combination increases risk of skin rash (maculopapular). Moderate interaction."),

    # Corticosteroids
    ("Prednisolone", "Ibuprofen",
     "Ibuprofen combined with prednisolone substantially increases GI bleed and ulcer risk. Major interaction."),
    ("Dexamethasone", "Warfarin",
     "Dexamethasone may increase or decrease warfarin anticoagulant effect. Unpredictable interaction — monitor INR."),
]


def create_database(db_path: str) -> None:
    """Create and seed the DrugBank lite SQLite database."""
    print(f"Creating database at: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS drugs (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            name      TEXT NOT NULL UNIQUE,
            indication TEXT DEFAULT '',
            status    TEXT DEFAULT 'approved',
            type      TEXT DEFAULT 'small molecule'
        );

        CREATE TABLE IF NOT EXISTS interactions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            drug1_name  TEXT NOT NULL,
            drug2_name  TEXT NOT NULL,
            description TEXT DEFAULT ''
        );

        CREATE INDEX IF NOT EXISTS idx_drugs_name       ON drugs(LOWER(name));
        CREATE INDEX IF NOT EXISTS idx_interactions_d1  ON interactions(LOWER(drug1_name));
        CREATE INDEX IF NOT EXISTS idx_interactions_d2  ON interactions(LOWER(drug2_name));
    """)

    cursor.executemany(
        "INSERT OR IGNORE INTO drugs (name, indication, status, type) VALUES (?, ?, ?, ?)",
        DRUGS,
    )

    for d1, d2, desc in INTERACTIONS:
        cursor.execute(
            "INSERT INTO interactions (drug1_name, drug2_name, description) VALUES (?, ?, ?)",
            (d1, d2, desc),
        )
        cursor.execute(
            "INSERT INTO interactions (drug1_name, drug2_name, description) VALUES (?, ?, ?)",
            (d2, d1, desc),
        )

    conn.commit()
    drug_count = cursor.execute("SELECT COUNT(*) FROM drugs").fetchone()[0]
    ix_count   = cursor.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
    conn.close()

    print(f"  ✓ {drug_count} drugs inserted")
    print(f"  ✓ {ix_count} interactions inserted (bidirectional)")
    print("Database ready.")


if __name__ == "__main__":
    create_database(DB_PATH)
