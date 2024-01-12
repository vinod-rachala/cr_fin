import csv
import random
from faker import Faker

fake = Faker()

# Define the number of records
num_records = 10000

# Define a list of labels: 0 for normal sentences, 1 for sentences containing PCI/PHI
labels = [0, 1]

# Function to generate synthetic PHI data
def generate_phi():
    return random.choice([fake.name(), fake.address(), fake.date_of_birth()])

# Function to generate synthetic PCI data
def generate_pci():
    return fake.credit_card_number()

# Open a file to write the data
with open('synthetic_pci_phi_dataset.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['sentence', 'label'])

    for _ in range(num_records):
        label = random.choice(labels)
        if label == 0:
            # Generate a random sentence without PCI/PHI data
            sentence = fake.sentence()
        else:
            # Generate a sentence with PCI/PHI data
            sensitive_data = random.choice([generate_pci(), generate_phi()])
            normal_sentence = fake.sentence()
            # Inject PCI/PHI data into a normal sentence
            sentence = normal_sentence.replace(random.choice(normal_sentence.split()), sensitive_data)

        writer.writerow([sentence, label])

print("Synthetic data generation complete.")
