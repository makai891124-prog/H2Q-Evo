class ArticleService
  def self.create_article(params, current_user)
    article = Article.new(params)
    article.user = current_user

    if article.save
      # Consider moving these to callbacks or observers for better separation of concerns
      create_tags(article, params[:tags]) if params[:tags]
      create_categories(article, params[:categories]) if params[:categories]
      
      Result.new(true, article, nil)
    else
      Result.new(false, nil, article.errors.full_messages)
    end
  end

  def self.update_article(article, params)
    if article.update(params)
      update_tags(article, params[:tags]) if params[:tags]
      update_categories(article, params[:categories]) if params[:categories]
      Result.new(true, article, nil)
    else
      Result.new(false, nil, article.errors.full_messages)
    end
  end

  def self.delete_article(article)
    if article.destroy
      Result.new(true, nil, nil)
    else
      Result.new(false, nil, article.errors.full_messages)
    end
  end

  private

  def self.create_tags(article, tag_names)
    tag_names.each do |tag_name|
      tag = Tag.find_or_create_by(name: tag_name)
      article.tags << tag unless article.tags.include?(tag)
    end
  end

  def self.update_tags(article, tag_names)
    # Remove existing tags
    article.tags.clear
    # Add new tags
    create_tags(article, tag_names)
  end

  def self.create_categories(article, category_names)
    category_names.each do |category_name
      category = Category.find_or_create_by(name: category_name)
      article.categories << category unless article.categories.include?(category)
    end
  end

  def self.update_categories(article, category_names)
    # Remove existing categories
    article.categories.clear
    # Add new categories
    create_categories(article, category_names)
  end

  class Result
    attr_reader :success, :data, :errors

    def initialize(success, data, errors)
      @success = success
      @data = data
      @errors = errors
    end
  end
end
