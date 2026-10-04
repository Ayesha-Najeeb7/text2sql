# Qualitative samples (dev)

## Correct

- **Q:** What position does the player who played for butler cc (ks) play?
  - gold: `SELECT Position FROM table WHERE School/Club Team = 'Butler CC (KS)'`
  - ours: `SELECT Position FROM table WHERE School/Club Team = 'butler cc (ks)'`
- **Q:** How many schools did player number 3 play at?
  - gold: `SELECT COUNT(School/Club Team) FROM table WHERE No. = 3`
  - ours: `SELECT COUNT(School/Club Team) FROM table WHERE No. = 3`
- **Q:** What school did player number 21 play for?
  - gold: `SELECT School/Club Team FROM table WHERE No. = 21`
  - ours: `SELECT School/Club Team FROM table WHERE No. = 21`
- **Q:** Who is the player that wears number 42?
  - gold: `SELECT Player FROM table WHERE No. = 42`
  - ours: `SELECT Player FROM table WHERE No. = 42`
- **Q:** What player played guard for toronto in 1996-97?
  - gold: `SELECT Player FROM table WHERE Position = 'Guard' AND Years in Toronto = '1996-97'`
  - ours: `SELECT Player FROM table WHERE Position = 'guard' AND Years in Toronto = '1996-97'`

## Wrong

- **Q:** Who are all of the players on the Westchester High School club team?
  - gold: `SELECT Player FROM table WHERE School/Club Team = 'Westchester High School'`
  - ours: `SELECT Player FROM table WHERE School/Club Team = 'westchester high'`
  - failure: **wrong value**
- **Q:** What school/club team is Amir Johnson on?
  - gold: `SELECT School/Club Team FROM table WHERE Player = 'Amir Johnson'`
  - ours: `SELECT School/Club Team FROM table WHERE Years in Toronto = 'amir johnson'`
  - failure: **wrong condition column/operator**
- **Q:** What school/club team is Trey Johnson on?
  - gold: `SELECT School/Club Team FROM table WHERE Player = 'Trey Johnson'`
  - ours: `SELECT School/Club Team FROM table WHERE Years in Toronto = 'trey johnson'`
  - failure: **wrong condition column/operator**
- **Q:** When did Jacques Chirac stop being a G8 leader?
  - gold: `SELECT Ended time as senior G8 leader FROM table WHERE Person = 'Jacques Chirac'`
  - ours: `SELECT Ended time as senior G8 leader FROM table WHERE Began time as senior G8 leader = 'jacques chirac stop'`
  - failure: **wrong condition column/operator**
- **Q:** When did the Prime Minister of Italy take office?
  - gold: `SELECT Entered office as Head of State or Government FROM table WHERE Office = 'Prime Minister of Italy'`
  - ours: `SELECT Office FROM table WHERE Began time as senior G8 leader = 'italy'`
  - failure: **wrong select column**
