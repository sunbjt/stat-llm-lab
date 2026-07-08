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
      res <- bpe_encode(self$model, x = text, type = "ids")[[1]]
      ids <- as.integer(res) + 1L 
      
      c(self$bos_idx, ids, self$eos_idx) 
    },
    
    encode_raw = function(text_vector) {
      res_list <- bpe_encode(self$model, x = text_vector, type = "ids")
      lapply(res_list, function(x) as.integer(x) + 1L)
    },
    
    decode = function(ids) {
      raw_ids <- as.integer(ids) - 1L
      bpe_decode(self$model, x = raw_ids)
    }
  )
)