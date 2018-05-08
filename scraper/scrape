#!/usr/bin/env ruby

require 'set'
require 'csv'
require 'fileutils'
require 'net/http'
require 'uri'
require 'open-uri'

require 'httparty'
require 'dotenv'

Dotenv.load('.env')

MASTER_IDS_FILE = "#{File.dirname(__FILE__)}/master_ids.txt"
MASTERS_DIR = "#{File.dirname(__FILE__)}/data"
META_FILE = "#{MASTERS_DIR}/metadata.csv"
SONGS_FILE = "#{MASTERS_DIR}/songs.csv"

class Master
  ATTRIBUTES  = %i(
    id
    title
    artist
    artist_id
    songs
    genres
    styles
    year
    primary_image
    secondary_image
  )

  attr_accessor *ATTRIBUTES

  def self.from_result(hash)
    master = new

    master.id = hash.fetch('id')
    master.title = hash.fetch('title')
    master.artist = hash['artists'].first['name'].gsub(/(\*|\(\d+\))$/, '')
    master.artist_id = hash['artists'].first['id']

    master.genres = hash['genres'].to_a.to_set
    master.styles = hash['styles'].to_a.to_set

    master.songs = hash['tracklist'].map { |t| t['title'] }.to_set
    master.year = hash['year']

    master.primary_image = hash['images'].
      find { |h| h['type'] == 'primary' }&.fetch('uri')

    master.secondary_image = hash['images'].
      find { |h| h['type'] == 'secondary' }&.fetch('uri')

    master
  end

  def dir
    "#{MASTERS_DIR}/#{artist_id}/#{id}"
  end
end

$master_ids = Set.new

class Client
  include HTTParty
  base_uri 'https://api.discogs.com'
  default_params key: ENV['DISCOGS_API_KEY'], secret: ENV['DISCOGS_API_SECRET']
  # headers('User-Agent' => 'lists.rathr.io')
  headers('User-Agent' => 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.13; rv:59.0) Gecko/20100101 Firefox/59.0')

  DEFAULT_PARAMS = [
    '/database/search',
    query: {
      per_page: 100,
      type: 'master'
    }
  ]

  def self.seed_master_ids(params: DEFAULT_PARAMS)
    response = get(*params)
    results = response['results'].to_a
    next_page = response.dig('pagination', 'urls', 'next')

    if results.empty?
      sleep(0.5)

      # Retry with Net::HTTP
      response = JSON.parse Net::HTTP.get(URI(params.first))
      results = response['results'].to_a
      next_page = response.dig('pagination', 'urls', 'next')

      if results.empty?
        require 'pry'; binding.pry
        return
      end
    end

    $master_ids = $master_ids.union(results.map { |r| r['id'] })
    print '.'

    sleep(0.5)
    seed_master_ids(params: [next_page])
  rescue => e
    require 'pry'; binding.pry
    puts
  end

  def self.master(id)
    response = get("/masters/#{id}")
    Master.from_result(response)
  end
end

def scrape_master_ids
  $master_ids = File.readlines(MASTER_IDS_FILE).map { |id| id.chomp.to_i }.to_set
  Client.seed_master_ids
ensure
  File.open(MASTER_IDS_FILE, 'w+') { |f| f.puts $master_ids.to_a }
  puts "\nWrote master IDs to #{MASTER_IDS_FILE}"
end

def generate_directory_structure(masters)
  masters.each do |master|
    FileUtils.mkdir_p master.dir
  end

  puts "Generated data directory structure at #{MASTERS_DIR}."
end

def generate_songs_csv(masters)
  CSV.open(SONGS_FILE, 'w') do |csv|
    csv << ['Artist', 'Artist ID', 'Album ID', 'Song']

    masters.each do |master|
      master.songs.each do |song|
        csv << [master.artist, master.artist_id, master.id, song]
      end
    end
  end

  puts "Generated #{SONGS_FILE}."
end

def generate_metadata_csv(masters)
  FileUtils.mkdir_p MASTERS_DIR

  CSV.open(META_FILE, 'w', col_sep: ';') do |csv|
    csv << ['ID', 'Title', 'Artist', 'Artist ID', 'Songs', 'Genres', 'Styles', 'Year', 'Primary Image', 'Secondary Image']

    masters.each do |master|
      csv << [
        master.id,
        master.title,
        master.artist,
        master.artist_id,
        master.songs.to_a.join('|'),
        master.genres.to_a.join('|'),
        master.styles.to_a.join('|'),
        master.year,
        master.primary_image,
        master.secondary_image
      ]

      # Download the pictures
      # begin
      #   if master.primary_image
      #     open(master.primary_image) do |f|
      #       File.open("#{master.dir}/primary.jpg", 'wb') { |file| file.puts f.read }
      #     end
      #   end

      #   if master.secondary_image
      #     open(master.secondary_image) do |f|
      #       File.open("#{master.dir}/secondary.jpg", 'wb') { |file| file.puts f.read }
      #     end
      #   end
      # rescue => e
      #   puts "Failed to download pics for #{master.title}: #{e.message}"
      # end
    end
  end

  puts "Generated #{META_FILE}."
end

$failed = Set.new
def scrape_masters
  masters = []

  master_ids = File.readlines(MASTER_IDS_FILE).map { |id| id.chomp.to_i }.to_set
  master_ids.each do |id|
    begin
      masters << Client.master(id)
      print '.'
      sleep(0.4)
    rescue
      print "x"
      $failed << id
      next
    end
  end

  masters
ensure
  puts
  # generate_directory_structure(masters)
  generate_metadata_csv(masters)
  # generate_songs_csv(masters)
end

# scrape_master_ids
scrape_masters

puts "\nDONE"
