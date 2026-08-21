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
          coverage = 0.999,
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
    
    encode = function(text) {
      punct_pattern <- "([,.:;!?\"'()\\{\\}\\[\\]，。！？；：—（）《》“”‘’、])"
      clean_text <- gsub(punct_pattern, " \\1 ", text, perl = TRUE)
      clean_text <- gsub("\\s+", " ", clean_text, perl = TRUE)

      res <- bpe_encode(self$model, x = clean_text, type = "ids")[[1]]
      ids <- as.integer(res) + 1L 
      
      c(self$bos_idx, ids, self$eos_idx) 
    },
    
    encode_raw = function(text_vector) {
      punct_pattern <- "([,.:;!?\"'()\\{\\}\\[\\]，。！？；：—（）《》“”‘’、])"
      clean_vector <- gsub(punct_pattern, " \\1 ", text_vector, perl = TRUE)
      clean_vector <- gsub("\\s+", " ", clean_vector, perl = TRUE)

      res_list <- bpe_encode(self$model, x = text_vector, type = "ids")
      lapply(res_list, function(x) as.integer(x) + 1L)
    },
    
    decode = function(ids, clean = TRUE) {
      raw_ids <- as.integer(ids) - 1L
      decoded <- bpe_decode(self$model, x = raw_ids)
      
      if (!clean) return(decoded)
      
      # 1. 过滤 Special Tokens
      decoded <- gsub("<BOS>|<EOS>|<PAD>|<UNK>", "", decoded, perl = TRUE)
      
      # 2. 清除 CJK 汉字与中文标点之间的空格（保留英文/数字之间的独立空格）
      cjk_pattern <- "(?<=[\\x{4e00}-\\x{9fa5}\\x{3000}-\\x{303f}\\x{ff00}-\\x{ffef}])\\s+(?=[\\x{4e00}-\\x{9fa5}\\x{3000}-\\x{303f}\\x{ff00}-\\x{ffef}])"
      decoded <- gsub(cjk_pattern, "", decoded, perl = TRUE)
      
      # 3. 修剪首尾空格
      trimws(decoded)
    }
  )
)