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

get '/:genre' do
  albums = Album.where(genre: params[:genre]).to_a.sample(400)
  erb :index, locals: { albums: albums }
end
