#!/usr/bin/env ruby

require 'sequel'
require 'csv'

DB = Sequel.sqlite('albums.db')

DB.create_table :albums do
  primary_key :id
  String :title
  String :artist
  Integer :artist_id
  String :songs
  String :genre
  String :genres
  String :styles
  Integer :year
  String :primary_image
  String :secondary_image
end

albums = DB[:albums]

CSV.foreach('../data/metadata.csv', col_sep: ';', headers: true) do |row|
  albums.insert(
    id: row['ID'],
    title: row['Title'],
    artist: row['Artist'],
    artist_id: row['Artist ID'],
    songs: row['Songs'],
    genre: row['Genres'].split('|').first,
    genres: row['Genres'],
    styles: row['Styles'],
    year: row['Year'],
    primary_image: row['Primary Image'],
    secondary_image: row['Secondary Image']
  )
end
