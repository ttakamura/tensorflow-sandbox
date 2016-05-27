# coding: utf-8
#
# tools02 などで下記を実行して TSV を準備する
#
# rails r 'Product.find_each{|pd| puts [pd.id, pd.main_image.url, (pd.categories||[]).join("|")].join("\t") }' | tee /tmp/products.tsv
# scp で持ってくる
#
require 'thread'

@queue = Queue.new

@threads = 10.times do
  Thread.start do
    loop do
      cmd = @queue.pop
      # puts cmd
      system(cmd)
    end
  end
end

DATA_DIR = 'data/tab_products/images'

products = {}

open(ARGV.shift).each_line do |line|
  id, url, categories = line.split("\t")
  if categories
    id         = id.to_i
    categories = categories.split("|")
    products[id] = {id: id, url: url, categories: categories}
  end
end

products.map do |key, prod|
  @queue << "curl -s '#{prod[:url]}' > #{DATA_DIR}/#{prod[:id]}.jpg"
end

until @queue.empty?
  p @queue.size
  sleep 1
end

#
# 下記コマンドなどで縮小する
#
# mogrify -quality 98 -type Grayscale -path data/tab_products/images_ss/ -resize 64x64 -gravity Center -background white -extent 64x64 data/tab_products/images_min/\*jpg
#
