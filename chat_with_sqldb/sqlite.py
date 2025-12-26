import sqlite3

### connect to sqlite
connection=sqlite3.connect('studentdb.db')
## create a cursor object to inseert record,create table
cursor=connection.cursor()

# create the table
table_info="""
    create table students(
    id integer primary key autoincrement,
    name varchar(100) not null,
    email varchar(100) unique not null,
    gender text check(gender in ('male','female','others')),
    date_of_birth date,
    created_at datetime default current_timestamp
    );
"""
cursor.execute(table_info)

## insert some more record
cursor.execute('''insert into students (id, name, email, gender, date_of_birth) values(1,'Ayan','ayanwork45@gmail.com','male','2004-07-19')''')
cursor.execute('''insert into students (id, name, email, gender, date_of_birth) values(2,'Sayan','sayanwork45@gmail.com','male','2005-09-29')''')
cursor.execute('''insert into students (id, name, email, gender, date_of_birth) values(3,'Ayantika','ayantikawork45@gmail.com','female','2003-03-29')''')

# display all the records
print('The inserted records are: ')
data=cursor.execute('''select * from students''')
for row in data:
    print(row)
# commmit my changes in databases
connection.commit()
connection.close()