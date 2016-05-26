
def name_to_id categories
  categories.map do |cat|
    unless @cat_master[cat]
      @cat_master[cat] = @cat_master.size
    end
    @cat_master[cat]
  end
end

data_dir = ARGV.shift
tsv_file = ARGV.shift

@cat_master = {}
products   = {}

open(tsv_file).each_line do |line|
  id, url, categories = line.chomp.split("\t")
  if categories
    categories = categories.split("|")
    if categories[2]
      categories = name_to_id(categories)
      products[id] = {categories: categories}
    end
  end
end

results = []

Dir.glob("#{data_dir}/*jpg").each do |file|
  id = file.split("/").last.gsub(".jpg","")
  if products[id]
    results << [products[id][:categories][1], products[id][:categories][2], id].join("\t")
  end
end

open("#{data_dir}/data.tsv", "w") do |file|
  results.each do |line|
    file.puts line
  end
end

open("#{data_dir}/category.tsv", "w") do |file|
  @cat_master.each do |name, id|
    file.puts [name, id].join("\t")
  end
end
