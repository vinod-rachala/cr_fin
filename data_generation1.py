from faker import Faker
import pandas as pd
import numpy as np

fake = Faker()

def create_pci_dataset(size):
    data = []
    for _ in range(size):
        record = {
            "credit_card_number": fake.credit_card_number(),
            "credit_card_provider": fake.credit_card_provider(),
            "name": fake.name(),
            "address": fake.address(),
            "email": fake.email(),
            "ssn": fake.ssn(),
            "phone_number": fake.phone_number(),
            "job": fake.job(),
            "birthdate": fake.date_of_birth(),
            "tax_id": fake.itin(),
            # ... add other fields as needed
        }
        # Add a sentence field that may or may not contain PCI data
        if np.random.rand() > 0.5:
            record["sentence"] = f"My credit card number is {record['credit_card_number']}"
            record["contains_pci"] = 1
        else:
            record["sentence"] = fake.sentence()
            record["contains_pci"] = 0
        data.append(record)
    
    return pd.DataFrame(data)

# Generate the dataset
df = create_pci_dataset(1000000)
