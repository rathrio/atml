Albums 
------

### Genre Distribution

* Electronic
* Rock
* Pop
* Jazz
* Hip Hop
* Funk / Soul
* Folk, World, & Country
* Classical
* Reggae
* Non-Music
* Latin
* Blues
* Children's
* Stage & Screen

### Location on the Server

The albums are stored at `/var/tmp/albums/data` on `node05`.
`/var/tmp/albums/data/metadata.csv` lists all 100'000+ albums with their
metadata, such as ID, artist, genres, etc.

The data directory itself then contains a folder for each album which contains a
folder for each artist of that album, which then contain the album covers. Note
that we call the albums "masters" in the metadata CSV, a naming we adopted from
the Discogs API. Should be the other way around but I messed it up in the
download script.


```sh
(/var/tmp/group5) [studi9@node05 albums]$ ./tree
data/
    metadata.csv                   # The metadata csv with all albums
    1095565/                       # Album ID because I messed up the order
        81131/                     # Artist ID
            primary.jpg            # Front Cover
    478444/
        2886362/
            primary.jpg
    176048/
        469595/
            secondary.jpg          # Optional Back Cover
            primary.jpg
    1019012/
        90235/
            secondary.jpg
            primary.jpg
    1198129/
        301147/
            primary.jpg
    673419/
        304779/
            secondary.jpg
            primary.jpg
```


Using the cluster
-----------------

First, make sure that you are in the Unibe network. Use a [VPN][0] if necessary.

1. ssh into the cluster (provide the password from the screenshot in the
   whatsapp group)

   ```
   $ ssh studi9@cluster.inf.unibe.ch
   ```
2. ssh into node05

   ```
   $ ssh node05
   ```
3. Load the correct python environment

   ```
   $ module load anaconda/3
   $ source activate /var/tmp/group5/
   ```

Done!

[0]: http://www.unibe.ch/university/campus_and_infrastructure/rund_um_computer/internetzugang/access_to_internal_resources_via_vpn/index_eng.html
