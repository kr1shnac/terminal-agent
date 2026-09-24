from memory import Memory


memory = Memory()

memory.add("User's name is Krishna", "fact")
memory.add("User is learning Python", "fact")
memory.add("User practices karate", "fact")

print("\nAll memories:")
print(memory.get_all())

print("\nSearch result:")
print(memory.search("What is my name?"))