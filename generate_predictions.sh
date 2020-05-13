for split in "test" "dev" "visual" "visual_easier" "situational_1" "situational_2" "contextual" "adverb_1" "adverb_2"
     do
     echo $split
     echo $1
     sbatch template.sbatch $split $1 $2
done 
#  1: /path/to/templates.s
# usage: generate_predctions.sh /path/to/checkpoint /path/to/output/dir
