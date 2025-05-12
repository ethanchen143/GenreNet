# 88 Genres Total
url_to_genre = [
    # GenreNetTraining1
    ("https://open.spotify.com/track/3SAga35lAPYdjj3qyfEsCF", "Alternative Pop"),
    ("https://open.spotify.com/track/6FhRpVWOtflqDk2UjWMY2N", "City Pop"),
    ("https://open.spotify.com/track/4E0P1xs3JNmsNr5c5nFTZJ", "Dream Pop"),
    ("https://open.spotify.com/track/4356Typ82hUiFAynbLYbPn", "Electropop"),
    ("https://open.spotify.com/track/2Nq4SFbvYYZa8AF7lD7CWU", "Indie Pop"),
    ("https://open.spotify.com/track/51ODNNDZm21HU7wI7cccRr", "Dance Pop"),
    ("https://open.spotify.com/track/7BMSBNctr9IPelr6MFvuRL", "Hyperpop"),
    ("https://open.spotify.com/track/6iqXxLX80xb2EAAzVNqewU", "Sunshine Pop"),
    ("https://open.spotify.com/track/71V89tJj9CboDyzncO6ZN2", "Bubblegum Pop"),
    ("https://open.spotify.com/track/7sMRDjjwsB7wQEBOkdfg0i", "J-Pop"),
    ("https://open.spotify.com/track/3coRPMnFg2dJcPu5RMloa9", "K-Pop"),
    ("https://open.spotify.com/track/72OVnXDzugvrCU25lMi9au", "C-Pop"),
    ("https://open.spotify.com/track/3kfaKqas6t8j07Ae1EB5fj", "Europop"),
    ("https://open.spotify.com/track/2Nq4SFbvYYZa8AF7lD7CWU", "Bedroom Pop"),
    ("https://open.spotify.com/track/5vmRQ3zELMLUQPo2FLQ76x", "Synth Pop"),
    ("https://open.spotify.com/track/78dbqYDvx6FOefROApu9w0", "Latin Pop"),
    ("https://open.spotify.com/track/2tH28YyKYOldxhuBHoI79M", "Yacht Rock"),
    ("https://open.spotify.com/track/4U45aEWtQhrm8A5mxPaFZ7", "Soft Rock"),
    ("https://open.spotify.com/track/60a0Rd6pjrkxjPbaKzXjfq", "Alternative Rock"),
    ("https://open.spotify.com/track/5e3YOg6fIkP0wD5TyxcHOH", "Garage Rock"),
    ("https://open.spotify.com/track/5XeFesFbtLpXzIVDNQP22n", "Indie Rock"),
    ("https://open.spotify.com/track/4fiOTntQKr24p07FvQDHZE", "Metal"),
    ("https://open.spotify.com/track/5XJ1J9QPxaOzdpkGxKU4lA", "New Wave"),
    ("https://open.spotify.com/track/5Ng6UbryNd3eds2zQk9MUf", "Post-Punk"),
    ("https://open.spotify.com/track/2wj1ZSQ32XsMKIvMNO61R2", "Progressive Rock"),
    ("https://open.spotify.com/track/14XWXWv5FoCbFzLksawpEe", "Psychedelic Rock"),
    ("https://open.spotify.com/track/56hwcJKj0M40A3qdhV3177", "Punk Rock"),
    ("https://open.spotify.com/track/33HRECrmuelZxOpid6XTNX", "Shoegaze"),
    ("https://open.spotify.com/track/3l9CW99AHtExIRV4hW2N5m", "Pop Punk"),
    ("https://open.spotify.com/track/4anUinKv803lyDD1vaSXhU", "Surf Rock"),
    ("https://open.spotify.com/track/7N3PAbqfTjSEU1edb2tY8j", "Hard Rock"),
    ("https://open.spotify.com/track/0hCB0YR03f6AmQaHbwWDe8", "Rock 'n' Roll"),
    ("https://open.spotify.com/track/5kAZ1jdxjYEqygmWugAwcF", "Grunge"),
    ("https://open.spotify.com/track/4R5qHNB2gdxBM2LDNaJZeO", "Glam Rock"),
    ("https://open.spotify.com/track/5yKA1w7RuOwQFbcve0Iukj", "Boom Bap"),
    ("https://open.spotify.com/track/5yY9lUy8nbvjM1Uyo1Uqoc", "Trap"),
    ("https://open.spotify.com/track/7IzutleLK1419FM8rUpYmq", "Rage"),
    ("https://open.spotify.com/track/0vMctOnb4YNIvbqgkbWNDy", "Jazz Rap"),
    ("https://open.spotify.com/track/3HkDOgyxSnO8XklAq8m0rX", "Trap Soul"),
    ("https://open.spotify.com/track/0aB0v4027ukVziUGwVGYpG", "Pop Rap"),
    ("https://open.spotify.com/track/3LtpKP5abr2qqjunvjlX5i", "Drill"),
    ("https://open.spotify.com/track/1zX178V8sWozr96MrfmRun", "Cloud Rap"),
    ("https://open.spotify.com/track/5YDUj8jdbeew1C6W5eT2IY", "G-Funk"),
    ("https://open.spotify.com/track/1RMJOxR6GRPsBHL8qeC2ux", "Contemporary R&B"),
    ("https://open.spotify.com/track/0tNuJpBgtE65diL6Q8Q7fI", "Neo Soul"),
    ("https://open.spotify.com/track/0n2pjCIMKwHSXoYfEbYMfX", "Soul"),
    ("https://open.spotify.com/track/0tjTndnyFm1xQsaHGf2imW", "Psychedelic Soul"),
    ("https://open.spotify.com/track/6ujbuB0L7JKxmpQzAoCMuK", "Slow Jams"),
    ("https://open.spotify.com/track/2MQ23MqDI0N9IVy8iaZddc", "Disco"),
    ("https://open.spotify.com/track/3HGE6Is63CrKHS9DZ26RIO", "New Jack Swing"),
    ("https://open.spotify.com/track/1uaGSDFsLdReQgg8p7Obwh", "Electronica"),
    ("https://open.spotify.com/track/4iLL2yVVG19TAJYssbMeBT", "Eurodance"),
    ("https://open.spotify.com/track/4WbNyQy4wVdfopMrApBWQA", "Future Bass"),
    ("https://open.spotify.com/track/2He3NOyqtLNE3RQPpeDdSb", "House"),
    ("https://open.spotify.com/track/4Ls53fBNVfaXTROBi6X8Hw", "Jersey Club"),
    ("https://open.spotify.com/track/5Cmxgp6kvf2M0HoSyjbfjA", "Nu Disco"),
    ("https://open.spotify.com/track/66wO3p3MofGc4yHSogrqqi", "Synthwave"),
    ("https://open.spotify.com/track/2kWB9IV8EHDOU9EjgxWFrF", "Techno"),
    ("https://open.spotify.com/track/7mPrEblYX8GLGRXLmPL8ws", "Trance"),
    ("https://open.spotify.com/track/2Np6ZbVFXfhlk3Oin6i92t", "UK Garage"),
    ("https://open.spotify.com/track/3dcWKFefG4Otjdb6ykBVcY", "Drum and Bass"),
    ("https://open.spotify.com/track/1KAg5YTwECiXmYeySVEpDS", "Dubstep"),
    ("https://open.spotify.com/track/6ASkheRUcaSa2TiP8fNPXB", "Hardstyle"),
    ("https://open.spotify.com/track/3dsFmdKYpgftTk6M22syBO", "Lo-Fi"),
    ("https://open.spotify.com/track/5em09RchDudiGPeweNEqpD", "Industrial"),
    ("https://open.spotify.com/track/1vgSaC0BPlL6LEm4Xsx59J", "Ambient"),
    # GenreNetTraining2
    ("https://open.spotify.com/track/0j5FJJOmmnXPd0XajFWkMF", "Trip Hop"),
    ("https://open.spotify.com/track/2rxQMGVafnNaRaXlRMWPde", "Country"),
    ("https://open.spotify.com/track/4gewzONBfahsWkWxaEhDDQ", "Bluegrass"),
    ("https://open.spotify.com/track/0iOZM63lendWRTTeKhZBSC", "Folk"),
    ("https://open.spotify.com/track/09hcbtRcZV5CeeygqQiM5f", "Cool Jazz"),
    ("https://open.spotify.com/track/1xicvSO4CJ2ymqYgpk7DFh", "Bebop"),
    ("https://open.spotify.com/track/4Oor2Emr3lDd3JtogJVzfC", "Jazz Fusion"),
    ("https://open.spotify.com/track/7vF5RIx2jHtt9Y0OElOZKK", "Gospel"),
    ("https://open.spotify.com/track/4NQfrmGs9iQXVQI9IpRhjM", "Blues"),
    ("https://open.spotify.com/track/65H6t1WQBim6q93yM8fEwn", "Bachata"),
    ("https://open.spotify.com/track/43558Td2trz7O0chZYohEE", "Corridos tumbados"),
    ("https://open.spotify.com/track/0Vl9aGb0dmeiCQ2ATgNK2B", "Bossa Nova"),
    ("https://open.spotify.com/track/6u0EAxf1OJTLS7CvInuNd7", "Baile Funk"),
    ("https://open.spotify.com/track/7yFvSYKk3g5g8e7Ffl16ws", "Reggae"),
    ("https://open.spotify.com/track/6IFDy0imCdhDpHj98GczEX", "Dancehall"),
    ("https://open.spotify.com/track/20pQ2kEbJnMO8nQx0oFfyg", "Afrobeats"),
    ("https://open.spotify.com/track/2ebzFs1h92yJBm7wOA39JO", "Amapiano"),
    ("https://open.spotify.com/track/2meEiZKWkiN28gITzFwQo5", "Pop"),
    ("https://open.spotify.com/track/0ECs7wpW9157Tk5yBUGbE0", "Hip Hop"),
    ("https://open.spotify.com/track/2V65y3PX4DkRhy1djlxd9p", "Electronic"),
    ("https://open.spotify.com/track/4iZ4pt7kvcaH6Yo8UoZ4s2", "R&B"),
    ("https://open.spotify.com/track/7znjbX9XdoQayIrVNdd50Z", "Jazz"),
    ("https://open.spotify.com/track/7znjbX9XdoQayIrVNdd50Z", "Jazz")
]

import pandas as pd

df = pd.read_csv("new_data.csv")

genre_labels = []
idx = -1
genre = ''

for curr in df['Spotify_URL']:
    url,curr_genre = url_to_genre[idx+1]
    if curr == url:
        idx += 1
        genre = curr_genre
    genre_labels.append(genre)

df['Ground_Truth_Genre'] = genre_labels
df.to_csv("labeled_data.csv", index=False)