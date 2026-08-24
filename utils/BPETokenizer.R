RtomicBPETokenizer <- R6Class(
  "RtomicBPETokenizer",
  public = list(
    model_path = NULL,
    model = NULL,
    vocab_size = NULL,
    
    # R 端的表层索引 (1-based)，绝不直接传给 C++
    pad_idx = 1L, unk_idx = 2L, bos_idx = 3L, eos_idx = 4L,
    
    initialize = function(corpus_file = NULL, model_file = "models/rtomic_bpe.model", vocab_size = 9000) {
      self$model_path <- model_file
      self$vocab_size <- vocab_size
      
      if (!file.exists(model_file)) {
        cat("正在调用 C++ 底层训练 BPE 模型 (这需要一点时间)...\n")
        self$model <- bpe(
          x = corpus_file, 
          model_path = model_file, 
          vocab_size = vocab_size,
          coverage = 0.9999,
          # C++ 底层使用标准的 0-based 索引
          pad_id = 0L, 
          unk_id = 1L, 
          bos_id = 2L, 
          eos_id = 3L
        )
        cat("BPE 训练完成！\n")
      } else {
        cat("检测到已存在的 BPE 模型，直接加载...\n")
        self$model <- bpe_load_model(model_file)
      }
    },
    
    # 私有辅助函数：中文友好型文本清洗
    clean_text_internal = function(text_vec) {
      # 1. 在所有中英文标点符号前后补空格，使标点在 BPE 统计时被当作独立单词切分
      punct_pattern <- "([,.:;!?\"'()\\{\\}\\[\\]，。！？；：—（）《》“”‘’、])"
      clean_vec <- gsub(punct_pattern, " \\1 ", text_vec, perl = TRUE)
      
      # 2. 合并多余空格
      gsub("\\s+", " ", clean_vec, perl = TRUE)
    },

    encode = function(text) {
      clean_text <- self$clean_text_internal(text)
      res <- bpe_encode(self$model, x = clean_text, type = "ids")[[1]]
      ids <- as.integer(res) + 1L 
      
      c(self$bos_idx, ids, self$eos_idx) 
    },
    
    encode_raw = function(text_vector) {
      clean_vector <- self$clean_text_internal(text_vector)
      res_list <- bpe_encode(self$model, x = text_vector, type = "ids")
      lapply(res_list, function(x) as.integer(x) + 1L)
    },
    
decode = function(ids, clean = TRUE) {
      raw_ids <- as.integer(ids) - 1L
      decoded <- bpe_decode(self$model, x = raw_ids)
      
      if (!clean) return(decoded)
      
      # 1. 过滤 Special Tokens
      decoded <- gsub("<BOS>|<EOS>|<PAD>|<UNK>", "", decoded, perl = TRUE)
      
      # 2. 精细还原：消除汉字与汉字、汉字与中文标点之间的所有空格
      # 前半段包含 CJK 字符/标点，后半段也包含 CJK 字符/标点
      cjk_pattern <- "(?<=[\\x{4e00}-\\x{9fa5}\\x{3000}-\\x{303f}\\x{ff00}-\\x{ffef}])\\s+(?=[\\x{4e00}-\\x{9fa5}\\x{3000}-\\x{303f}\\x{ff00}-\\x{ffef}])"
      decoded <- gsub(cjk_pattern, "", decoded, perl = TRUE)
      
      # 清理英文/数字与半角标点的多余空格
      decoded <- gsub("(?<=[a-zA-Z0-9])\\s+(?=[,.?!:;/()])", "", decoded, perl = TRUE)
      decoded <- gsub("(?<=[(.])\\s+(?=[a-zA-Z0-9])", "", decoded, perl = TRUE)
      
      trimws(decoded)
    }
  )
)