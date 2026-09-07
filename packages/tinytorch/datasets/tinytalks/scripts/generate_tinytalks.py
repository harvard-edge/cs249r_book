"""
TinyTalks Dataset Generator

Systematically generates the TinyTalks v1.0 dataset with 350 Q&A pairs
across 5 difficulty levels.

Usage:
    python scripts/generate_tinytalks.py

Output:
    - tinytalks_v1.txt (full dataset)
    - splits/train.txt (70%)
    - splits/val.txt (15%)
    - splits/test.txt (15%)
"""

import random
from pathlib import Path


def generate_level1_greetings():
    """Level 1: Greetings & Identity (50 pairs)"""
    qa_pairs = [
        # Basic greetings
        ("Hello!", "Hi there! How can I help you today?"),
        ("Hi!", "Hello! What can I do for you?"),
        ("Hey!", "Hey! Nice to meet you."),
        ("Good morning!", "Good morning! How are you today?"),
        ("Good afternoon!", "Good afternoon! How can I assist you?"),
        ("Good evening!", "Good evening! What brings you here?"),
        
        # Identity questions
        ("What is your name?", "I am TinyBot, a simple AI assistant."),
        ("Who are you?", "I am TinyBot, here to help answer questions."),
        ("What are you?", "I am an AI assistant called TinyBot."),
        ("Are you a robot?", "Yes, I am an AI assistant."),
        ("Are you human?", "No, I am an AI assistant, not a human."),
        
        # Capability questions
        ("What can you do?", "I can answer questions and have simple conversations."),
        ("Can you help me?", "Yes, I can try to help you with questions."),
        ("How are you?", "I am doing well, thank you for asking!"),
        ("How do you work?", "I am a language model that processes text."),
        
        # Politeness
        ("Thank you!", "You are welcome!"),
        ("Thanks!", "My pleasure!"),
        ("You are helpful!", "Thank you, I am happy to help!"),
        ("Goodbye!", "Goodbye! Have a great day!"),
        ("Bye!", "Bye! See you later!"),
        ("See you later!", "See you! Take care!"),
        
        # Simple questions about TinyBot
        ("Where are you from?", "I was created by the TinyTorch team."),
        ("When were you made?", "I was created as part of the TinyTorch project."),
        ("Why were you created?", "I was created to help teach transformer models."),
        
        # Simple confirmations
        ("Are you there?", "Yes, I am here!"),
        ("Can you hear me?", "Yes, I can read your messages."),
        ("Do you understand?", "Yes, I understand your question."),
        
        # Feelings (simple)
        ("Are you happy?", "I do not have feelings, but I am here to help!"),
        ("Do you like helping?", "I am designed to help, so yes!"),
        
        # Meta questions
        ("What language do you speak?", "I speak English."),
        ("Can you speak Spanish?", "I primarily speak English."),
        ("Do you know other languages?", "I am focused on English right now."),
        
        # Introductions
        ("My name is Alice.", "Nice to meet you, Alice!"),
        ("I am Bob.", "Hello Bob, I am TinyBot!"),
        ("Call me Charlie.", "Nice to meet you, Charlie!"),
        
        # Simple requests
        ("Tell me about yourself.", "I am TinyBot, a simple AI assistant for answering questions."),
        ("Introduce yourself.", "Hello, I am TinyBot, here to help you!"),
        ("Say hello.", "Hello! How are you?"),
        
        # More greetings
        ("Greetings!", "Greetings! What can I do for you?"),
        ("Hi TinyBot!", "Hi! How can I help you today?"),
        ("Hello TinyBot!", "Hello! I am here to assist you."),
        ("Hey there!", "Hey! What would you like to know?"),
        
        # Questions about the conversation
        ("Can we talk?", "Yes, I am ready to talk!"),
        ("Can you chat?", "Yes, I can chat with you!"),
        ("Do you want to talk?", "Sure, I am here to talk!"),
        
        # Simple acknowledgments
        ("Okay.", "Alright! What else can I help with?"),
        ("I see.", "Great! Any other questions?"),
    ]
    return qa_pairs


def generate_level2_simple_facts():
    """Level 2: Simple Facts (100 pairs)"""
    qa_pairs = [
        # Colors
        ("What color is the sky?", "The sky is blue during the day."),
        ("What color is grass?", "Grass is green."),
        ("What color is the sun?", "The sun is yellow."),
        ("What color is snow?", "Snow is white."),
        ("What color are clouds?", "Clouds are usually white or gray."),
        ("What color is an apple?", "Apples can be red, green, or yellow."),
        ("What color is a banana?", "A banana is yellow."),
        ("What color are leaves?", "Leaves are usually green."),
        ("What color is the ocean?", "The ocean is blue."),
        ("What color is fire?", "Fire is red, orange, and yellow."),
        
        # Animals - body parts
        ("How many legs does a dog have?", "A dog has four legs."),
        ("How many legs does a cat have?", "A cat has four legs."),
        ("How many legs does a bird have?", "A bird has two legs."),
        ("How many legs does a spider have?", "A spider has eight legs."),
        ("How many legs does an ant have?", "An ant has six legs."),
        ("How many wings does a bird have?", "A bird has two wings."),
        ("How many eyes do people have?", "People have two eyes."),
        ("How many ears does a cat have?", "A cat has two ears."),
        
        # Animals - sounds
        ("What sound does a dog make?", "A dog barks or woofs."),
        ("What sound does a cat make?", "A cat meows or purrs."),
        ("What sound does a cow make?", "A cow moos."),
        ("What sound does a bird make?", "A bird chirps or tweets."),
        ("What sound does a lion make?", "A lion roars."),
        ("What sound does a sheep make?", "A sheep baahs."),
        
        # Numbers and time
        ("How many days in a week?", "There are seven days in a week."),
        ("How many months in a year?", "There are twelve months in a year."),
        ("How many hours in a day?", "There are twenty-four hours in a day."),
        ("How many minutes in an hour?", "There are sixty minutes in an hour."),
        ("How many seconds in a minute?", "There are sixty seconds in a minute."),
        
        # Days of the week
        ("What is the first day of the week?", "Monday is the first day of the week."),
        ("What is the last day of the week?", "Sunday is the last day of the week."),
        ("What day comes after Monday?", "Tuesday comes after Monday."),
        ("What day comes before Sunday?", "Saturday comes before Sunday."),
        
        # Seasons
        ("What are the four seasons?", "The four seasons are spring, summer, fall, and winter."),
        ("When is it hot?", "It is hot in summer."),
        ("When is it cold?", "It is cold in winter."),
        ("When do leaves fall?", "Leaves fall in autumn or fall."),
        ("When do flowers bloom?", "Flowers bloom in spring."),
        
        # Basic geography
        ("What is the capital of France?", "The capital of France is Paris."),
        ("What is the capital of England?", "The capital of England is London."),
        ("What is the largest ocean?", "The Pacific Ocean is the largest ocean."),
        ("What is the tallest mountain?", "Mount Everest is the tallest mountain."),
        
        # Planets
        ("How many planets are there?", "There are eight planets in our solar system."),
        ("What is the closest planet to the sun?", "Mercury is the closest planet to the sun."),
        ("What is the largest planet?", "Jupiter is the largest planet."),
        ("What planet do we live on?", "We live on planet Earth."),
        
        # Body parts
        ("What do you see with?", "You see with your eyes."),
        ("What do you hear with?", "You hear with your ears."),
        ("What do you smell with?", "You smell with your nose."),
        ("What do you taste with?", "You taste with your tongue."),
        ("What do you walk with?", "You walk with your legs and feet."),
        
        # Food basics
        ("What do bees make?", "Bees make honey."),
        ("What do chickens lay?", "Chickens lay eggs."),
        ("What do cows give?", "Cows give milk."),
        ("Where does bread come from?", "Bread comes from wheat flour."),
        ("Where does cheese come from?", "Cheese comes from milk."),
        
        # Objects and materials
        ("What is paper made from?", "Paper is made from trees."),
        ("What is glass made from?", "Glass is made from sand."),
        ("What is a chair made from?", "A chair can be made from wood or metal."),
        ("What is water made of?", "Water is made of hydrogen and oxygen."),
        
        # Simple true/false facts
        ("Is the sun a star?", "Yes, the sun is a star."),
        ("Is water wet?", "Yes, water is wet."),
        ("Can birds fly?", "Yes, most birds can fly."),
        ("Can fish breathe underwater?", "Yes, fish breathe underwater using gills."),
        ("Do plants need sunlight?", "Yes, plants need sunlight to grow."),
        ("Do humans need water?", "Yes, humans need water to survive."),
        ("Is ice cold?", "Yes, ice is cold."),
        ("Is fire hot?", "Yes, fire is hot."),
        
        # Shapes
        ("How many sides does a triangle have?", "A triangle has three sides."),
        ("How many sides does a square have?", "A square has four sides."),
        ("How many sides does a circle have?", "A circle has no sides."),
        ("What shape is a ball?", "A ball is a sphere."),
        
        # More animal facts
        ("Where do fish live?", "Fish live in water."),
        ("Where do birds live?", "Birds live in nests or trees."),
        ("What do cats eat?", "Cats eat meat or cat food."),
        ("What do cows eat?", "Cows eat grass and hay."),
        ("What do bees eat?", "Bees eat nectar and pollen."),
        
        # More basic facts
        ("What is the opposite of hot?", "The opposite of hot is cold."),
        ("What is the opposite of big?", "The opposite of big is small."),
        ("What is the opposite of day?", "The opposite of day is night."),
        ("What is the opposite of up?", "The opposite of up is down."),
        ("What is the opposite of fast?", "The opposite of fast is slow."),
    ]
    return qa_pairs


def generate_level3_basic_math():
    """Level 3: Basic Math (50 pairs)"""
    qa_pairs = [
        # Addition (single digit)
        ("What is 1 plus 1?", "1 plus 1 equals 2."),
        ("What is 2 plus 2?", "2 plus 2 equals 4."),
        ("What is 2 plus 3?", "2 plus 3 equals 5."),
        ("What is 3 plus 4?", "3 plus 4 equals 7."),
        ("What is 4 plus 5?", "4 plus 5 equals 9."),
        ("What is 5 plus 5?", "5 plus 5 equals 10."),
        ("What is 6 plus 3?", "6 plus 3 equals 9."),
        ("What is 7 plus 2?", "7 plus 2 equals 9."),
        ("What is 8 plus 1?", "8 plus 1 equals 9."),
        ("What is 3 plus 3?", "3 plus 3 equals 6."),
        ("What is 4 plus 4?", "4 plus 4 equals 8."),
        ("What is 6 plus 6?", "6 plus 6 equals 12."),
        ("What is 7 plus 7?", "7 plus 7 equals 14."),
        ("What is 8 plus 8?", "8 plus 8 equals 16."),
        ("What is 9 plus 9?", "9 plus 9 equals 18."),
        
        # Subtraction (single digit)
        ("What is 5 minus 2?", "5 minus 2 equals 3."),
        ("What is 7 minus 3?", "7 minus 3 equals 4."),
        ("What is 9 minus 4?", "9 minus 4 equals 5."),
        ("What is 8 minus 5?", "8 minus 5 equals 3."),
        ("What is 10 minus 6?", "10 minus 6 equals 4."),
        ("What is 6 minus 2?", "6 minus 2 equals 4."),
        ("What is 9 minus 5?", "9 minus 5 equals 4."),
        ("What is 7 minus 4?", "7 minus 4 equals 3."),
        ("What is 8 minus 3?", "8 minus 3 equals 5."),
        ("What is 10 minus 5?", "10 minus 5 equals 5."),
        
        # Multiplication (simple)
        ("What is 2 times 2?", "2 times 2 equals 4."),
        ("What is 3 times 3?", "3 times 3 equals 9."),
        ("What is 4 times 2?", "4 times 2 equals 8."),
        ("What is 5 times 2?", "5 times 2 equals 10."),
        ("What is 2 times 5?", "2 times 5 equals 10."),
        ("What is 3 times 2?", "3 times 2 equals 6."),
        ("What is 2 times 4?", "2 times 4 equals 8."),
        ("What is 6 times 2?", "6 times 2 equals 12."),
        ("What is 2 times 7?", "2 times 7 equals 14."),
        ("What is 5 times 3?", "5 times 3 equals 15."),
        
        # Division (simple)
        ("What is 10 divided by 2?", "10 divided by 2 equals 5."),
        ("What is 8 divided by 2?", "8 divided by 2 equals 4."),
        ("What is 6 divided by 2?", "6 divided by 2 equals 3."),
        ("What is 12 divided by 3?", "12 divided by 3 equals 4."),
        ("What is 15 divided by 3?", "15 divided by 3 equals 5."),
        
        # Comparisons
        ("Which is bigger, 5 or 3?", "5 is bigger than 3."),
        ("Which is smaller, 2 or 7?", "2 is smaller than 7."),
        ("Which is larger, 10 or 8?", "10 is larger than 8."),
        ("Which is less, 4 or 6?", "4 is less than 6."),
        ("Is 9 greater than 7?", "Yes, 9 is greater than 7."),
    ]
    return qa_pairs


def generate_level4_reasoning():
    """Level 4: Common Sense Reasoning (100 pairs)"""
    qa_pairs = [
        # Object purposes
        ("What do you use a pen for?", "You use a pen to write."),
        ("What do you use scissors for?", "You use scissors to cut things."),
        ("What do you use a hammer for?", "You use a hammer to hit nails."),
        ("What do you use an umbrella for?", "You use an umbrella to stay dry in the rain."),
        ("What do you use a spoon for?", "You use a spoon to eat soup or cereal."),
        ("What do you use a key for?", "You use a key to open a lock."),
        ("What do you use a phone for?", "You use a phone to make calls or send messages."),
        ("What do you use a computer for?", "You use a computer to work or browse the internet."),
        ("What do you use a toothbrush for?", "You use a toothbrush to clean your teeth."),
        ("What do you use soap for?", "You use soap to wash and clean."),
        
        # Locations
        ("Where do you sleep?", "You sleep in a bed."),
        ("Where do you eat?", "You eat at a table."),
        ("Where do you cook?", "You cook in a kitchen."),
        ("Where do you study?", "You study at a desk or table."),
        ("Where do you swim?", "You swim in a pool or ocean."),
        ("Where do you buy food?", "You buy food at a store or market."),
        ("Where do you see a doctor?", "You see a doctor at a hospital or clinic."),
        ("Where do you learn?", "You learn at school."),
        ("Where do you watch movies?", "You watch movies at a theater or at home."),
        ("Where do books live?", "Books live on shelves or in libraries."),
        
        # Cause and effect
        ("What happens if you drop water?", "If you drop water, it spills."),
        ("What happens if you heat ice?", "If you heat ice, it melts."),
        ("What happens if you plant a seed?", "If you plant a seed, it grows into a plant."),
        ("What happens when it rains?", "When it rains, things get wet."),
        ("What happens if you turn off the light?", "If you turn off the light, it gets dark."),
        ("What happens if you eat too much?", "If you eat too much, you feel full."),
        ("What happens if you do not sleep?", "If you do not sleep, you feel tired."),
        ("What happens if you exercise?", "If you exercise, you get stronger."),
        ("What happens when you smile?", "When you smile, people think you are happy."),
        ("What happens if you study hard?", "If you study hard, you learn more."),
        
        # Physical properties
        ("Is a rock hard or soft?", "A rock is hard."),
        ("Is a pillow hard or soft?", "A pillow is soft."),
        ("Is metal heavy or light?", "Metal is usually heavy."),
        ("Is a feather heavy or light?", "A feather is light."),
        ("Is ice cream hot or cold?", "Ice cream is cold."),
        ("Is coffee hot or cold?", "Coffee is usually hot."),
        
        # Time and sequence
        ("When do you eat breakfast?", "You eat breakfast in the morning."),
        ("When do you eat lunch?", "You eat lunch at noon or midday."),
        ("When do you eat dinner?", "You eat dinner in the evening."),
        ("When do you go to bed?", "You go to bed at night."),
        ("When do you wake up?", "You wake up in the morning."),
        ("What comes after Monday?", "Tuesday comes after Monday."),
        ("What comes before Friday?", "Thursday comes before Friday."),
        ("What season comes after winter?", "Spring comes after winter."),
        
        # Needs and requirements
        ("What do plants need to grow?", "Plants need water, sunlight, and soil to grow."),
        ("What do people need to live?", "People need food, water, and air to live."),
        ("What do cars need to run?", "Cars need fuel or electricity to run."),
        ("What do fires need to burn?", "Fires need oxygen, fuel, and heat to burn."),
        
        # Logical relationships
        ("If it is raining, what should you bring?", "If it is raining, you should bring an umbrella."),
        ("If you are hungry, what should you do?", "If you are hungry, you should eat food."),
        ("If you are thirsty, what should you drink?", "If you are thirsty, you should drink water."),
        ("If you are cold, what should you wear?", "If you are cold, you should wear a jacket or coat."),
        ("If you are tired, what should you do?", "If you are tired, you should rest or sleep."),
        ("If the light is green, what should you do?", "If the light is green, you should go."),
        ("If the light is red, what should you do?", "If the light is red, you should stop."),
        
        # Categories
        ("Is a dog an animal?", "Yes, a dog is an animal."),
        ("Is an apple a fruit?", "Yes, an apple is a fruit."),
        ("Is a carrot a vegetable?", "Yes, a carrot is a vegetable."),
        ("Is a chair furniture?", "Yes, a chair is furniture."),
        ("Is a car a vehicle?", "Yes, a car is a vehicle."),
        ("Is water a liquid?", "Yes, water is a liquid."),
        ("Is ice a solid?", "Yes, ice is a solid."),
        
        # Simple reasoning
        ("Can you eat a book?", "No, you cannot eat a book."),
        ("Can you drink juice?", "Yes, you can drink juice."),
        ("Can you sit on a chair?", "Yes, you can sit on a chair."),
        ("Can a fish fly?", "No, a fish cannot fly."),
        ("Can a plane fly?", "Yes, a plane can fly."),
        ("Can you swim in milk?", "Technically yes, but it is not normal."),
        ("Do trees have roots?", "Yes, trees have roots."),
        ("Do fish have legs?", "No, fish do not have legs."),
        ("Do birds have feathers?", "Yes, birds have feathers."),
        ("Do mammals have fur or hair?", "Yes, most mammals have fur or hair."),
        
        # Emotions and feelings
        ("What makes you happy?", "Good things make people happy."),
        ("What makes you sad?", "Bad things make people sad."),
        ("How do you feel when you win?", "You feel happy when you win."),
        ("How do you feel when you lose?", "You feel sad when you lose."),
        
        # Weather
        ("What do you wear when it is cold?", "You wear a coat when it is cold."),
        ("What do you wear when it is hot?", "You wear light clothes when it is hot."),
        ("What do you use when it is sunny?", "You use sunglasses or sunscreen when it is sunny."),
        
        # More object uses
        ("What do you write on?", "You write on paper."),
        ("What do you read?", "You read books or articles."),
        ("What do you listen to?", "You listen to music or sounds."),
        ("What do you watch?", "You watch TV, movies, or videos."),
        
        # More locations
        ("Where do you park a car?", "You park a car in a parking lot or garage."),
        ("Where do you catch a train?", "You catch a train at a train station."),
        ("Where do you fly from?", "You fly from an airport."),
        ("Where do you mail letters?", "You mail letters at a post office."),
    ]
    return qa_pairs


def generate_level5_context():
    """Level 5: Multi-turn Context (50 pairs)"""
    qa_pairs = [
        # Food preferences
        ("I like pizza.", "Pizza is delicious! What toppings do you like?"),
        ("I like pepperoni.", "Pepperoni is a popular choice!"),
        ("Do you like cheese?", "I am an AI, so I do not eat, but cheese is popular!"),
        
        # Activities
        ("I went to the park.", "That sounds fun! What did you do there?"),
        ("I played soccer.", "Soccer is a great sport!"),
        ("Do you play sports?", "I am an AI, so I do not play sports."),
        
        # Weather
        ("It is sunny today.", "Sunny weather is nice! Are you going outside?"),
        ("Yes, I am going to the beach.", "The beach sounds wonderful on a sunny day!"),
        
        # School
        ("I go to school.", "School is important! What is your favorite subject?"),
        ("I like math.", "Math is a useful subject!"),
        ("Do you like science?", "I think science is fascinating!"),
        
        # Animals
        ("I have a dog.", "Dogs are wonderful pets! What is your dog's name?"),
        ("His name is Max.", "Max is a great name for a dog!"),
        ("Do you have pets?", "I am an AI, so I do not have pets."),
        
        # Hobbies
        ("I like reading.", "Reading is a great hobby! What do you like to read?"),
        ("I like adventure books.", "Adventure books are exciting!"),
        
        # Travel
        ("I visited Paris.", "Paris is a beautiful city! Did you see the Eiffel Tower?"),
        ("Yes, I did!", "The Eiffel Tower is amazing!"),
        
        # Colors
        ("My favorite color is blue.", "Blue is a lovely color!"),
        ("What is your favorite color?", "I do not have preferences, but blue is popular!"),
        
        # Time
        ("It is morning.", "Good morning! How are you today?"),
        ("I am fine.", "That is great to hear!"),
        
        # Family
        ("I have a brother.", "That is nice! Do you have other siblings?"),
        ("I also have a sister.", "So you have two siblings!"),
        
        # Learning
        ("I am learning English.", "English is a useful language! How is it going?"),
        ("It is going well.", "That is wonderful! Keep practicing!"),
        
        # Music
        ("I like music.", "Music is wonderful! What kind of music do you like?"),
        ("I like pop music.", "Pop music is very popular!"),
        
        # Games
        ("I play video games.", "Video games are fun! What games do you play?"),
        ("I play puzzle games.", "Puzzle games are great for the brain!"),
        
        # Feelings
        ("I am happy today.", "That is wonderful! What made you happy?"),
        ("I got a good grade.", "Congratulations on your good grade!"),
        
        # Plans
        ("I am going shopping.", "Shopping can be fun! What are you buying?"),
        ("I need new shoes.", "Finding good shoes is important!"),
        
        # Technology
        ("I have a new phone.", "New phones are exciting! Do you like it?"),
        ("Yes, it is very fast.", "Fast phones make everything easier!"),
        
        # Birthday
        ("My birthday is tomorrow.", "Happy early birthday! How old will you be?"),
        ("I will be ten.", "Ten is a great age!"),
        
        # Movies
        ("I saw a movie.", "Movies are entertaining! What movie did you see?"),
        ("I saw an action movie.", "Action movies are exciting!"),
    ]
    return qa_pairs


def create_dataset():
    """Generate the complete TinyTalks dataset"""
    print("Generating TinyTalks v1.0 dataset...")
    
    # Generate all levels
    level1 = generate_level1_greetings()
    level2 = generate_level2_simple_facts()
    level3 = generate_level3_basic_math()
    level4 = generate_level4_reasoning()
    level5 = generate_level5_context()
    
    # Combine all Q&A pairs with level tags
    all_pairs = []
    all_pairs.extend([("L1", q, a) for q, a in level1])
    all_pairs.extend([("L2", q, a) for q, a in level2])
    all_pairs.extend([("L3", q, a) for q, a in level3])
    all_pairs.extend([("L4", q, a) for q, a in level4])
    all_pairs.extend([("L5", q, a) for q, a in level5])
    
    print(f"Total Q&A pairs: {len(all_pairs)}")
    print(f"  Level 1 (Greetings): {len(level1)}")
    print(f"  Level 2 (Facts): {len(level2)}")
    print(f"  Level 3 (Math): {len(level3)}")
    print(f"  Level 4 (Reasoning): {len(level4)}")
    print(f"  Level 5 (Context): {len(level5)}")
    
    # Set seed for reproducible splits
    random.seed(42)
    random.shuffle(all_pairs)
    
    # Split into train/val/test (70/15/15)
    total = len(all_pairs)
    train_size = int(0.70 * total)
    val_size = int(0.15 * total)
    
    train_pairs = all_pairs[:train_size]
    val_pairs = all_pairs[train_size:train_size + val_size]
    test_pairs = all_pairs[train_size + val_size:]
    
    print(f"\nSplits:")
    print(f"  Train: {len(train_pairs)} ({len(train_pairs)/total*100:.1f}%)")
    print(f"  Val: {len(val_pairs)} ({len(val_pairs)/total*100:.1f}%)")
    print(f"  Test: {len(test_pairs)} ({len(test_pairs)/total*100:.1f}%)")
    
    return all_pairs, train_pairs, val_pairs, test_pairs


def format_qa_pairs(pairs, include_level_tags=False):
    """Format Q&A pairs as text"""
    lines = []
    for item in pairs:
        if include_level_tags:
            level, q, a = item
            lines.append(f"Q: {q}")
            lines.append(f"A: {a}")
            lines.append("")  # Empty line separator
        else:
            q, a = item
            lines.append(f"Q: {q}")
            lines.append(f"A: {a}")
            lines.append("")  # Empty line separator
    return "\n".join(lines)


def save_dataset(all_pairs, train_pairs, val_pairs, test_pairs):
    """Save dataset files"""
    script_dir = Path(__file__).parent
    dataset_dir = script_dir.parent
    splits_dir = dataset_dir / "splits"
    splits_dir.mkdir(exist_ok=True)
    
    print("\nSaving files...")
    
    # Save full dataset (with level tags for reference)
    full_path = dataset_dir / "tinytalks_v1.txt"
    with open(full_path, 'w', encoding='utf-8') as f:
        f.write(format_qa_pairs([(q, a) for _, q, a in all_pairs]))
    print(f"  ✓ {full_path}")
    
    # Save splits (without level tags)
    train_path = splits_dir / "train.txt"
    with open(train_path, 'w', encoding='utf-8') as f:
        f.write(format_qa_pairs([(q, a) for _, q, a in train_pairs]))
    print(f"  ✓ {train_path}")
    
    val_path = splits_dir / "val.txt"
    with open(val_path, 'w', encoding='utf-8') as f:
        f.write(format_qa_pairs([(q, a) for _, q, a in val_pairs]))
    print(f"  ✓ {val_path}")
    
    test_path = splits_dir / "test.txt"
    with open(test_path, 'w', encoding='utf-8') as f:
        f.write(format_qa_pairs([(q, a) for _, q, a in test_pairs]))
    print(f"  ✓ {test_path}")
    
    print("\n✅ TinyTalks v1.0 dataset generated successfully!")
    
    # Print file sizes
    print(f"\nFile sizes:")
    print(f"  Full dataset: {full_path.stat().st_size / 1024:.1f} KB")
    print(f"  Train split: {train_path.stat().st_size / 1024:.1f} KB")
    print(f"  Val split: {val_path.stat().st_size / 1024:.1f} KB")
    print(f"  Test split: {test_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    all_pairs, train_pairs, val_pairs, test_pairs = create_dataset()
    save_dataset(all_pairs, train_pairs, val_pairs, test_pairs)

