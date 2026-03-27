import sqlite3

# connect to the database
conn = sqlite3.connect("users.db")
cursor = conn.cursor()

# username to delete
username = "mohammed"

# delete user
cursor.execute("DELETE FROM users WHERE username = ?", (username,))

# save changes
conn.commit()

print("User deleted if it existed.")

# close connection
conn.close()