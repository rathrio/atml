# frozen_string_literal: true

require 'sinatra'
require 'sequel'

DB = Sequel.sqlite('albums.db')

class Album < Sequel::Model(DB[:albums])
  def image
    return secondary_image if primary_image.nil?
    primary_image
  end
end

get '/' do
  albums = Album

  if params[:genre].nil? && params[:artist].nil?
    return "No filters applied"
  end

  if params[:genre]
    albums = albums.where(genre: params[:genre])
  end

  if params[:artist]
    albums = albums.where(Sequel.like(:artist, "%#{params[:artist]}%"))
  end

  erb :index, locals: { albums: albums.to_a.sample(80) }
end
