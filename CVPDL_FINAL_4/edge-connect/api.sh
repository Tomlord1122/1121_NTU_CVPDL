# # inference
# python test.py \
#   --checkpoints ./checkpoints/places2 \
#   --input ./examples/places2/images \
#   --mask ./examples/places2/masks \
#   --output ./checkpoints/results 

python test.py \
  --checkpoints ./checkpoints/places2 \
  --input ./examples/tom/images \
  --mask ./examples/tom/masks \
  --output ./checkpoints/results 
  