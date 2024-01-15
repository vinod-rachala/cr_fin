from faker import Faker
import pandas as pd
import numpy as np

fake = Faker()

def create_dataset(size):
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
            # ... other fields ...
            "sentence": fake.sentence(),
            "contains_sensitive_data": np.random.randint(0, 2)
        }
        data.append(record)

    return pd.DataFrame(data)

df = create_dataset(1000000)
