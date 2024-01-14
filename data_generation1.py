from faker import Faker
import pandas as pd
import numpy as np

fake = Faker()

def create_synthetic_data(size):
    data = []
    for _ in range(size):
        record = {
            "credit_card_number": fake.credit_card_number(),
            "name": fake.name(),
            "address": fake.address(),
            "email": fake.email(),
            "ssn": fake.ssn(),
            "phone_number": fake.phone_number(),
            "birthdate": fake.date_of_birth(),
            "medical_info": fake.text(),
            "tax_id": fake.itin(),
            # ... other fields
            "sentence": "",
            "contains_pci_phi_pii": 0
        }

        if np.random.rand() > 0.5:
            # Include PCI, PHI, or PII in the sentence
            choice = np.random.choice(["credit_card_number", "ssn", "medical_info"])
            record["sentence"] = f"My {choice} is {record[choice]}"
            record["contains_pci_phi_pii"] = 1
        else:
            record["sentence"] = fake.sentence()

        data.append(record)

    return pd.DataFrame(data)

# Generate the dataset
df = create_synthetic_data(1000000)
