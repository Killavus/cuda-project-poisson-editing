#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <argp.h>

#include "include/host_image.h"
#include "include/naive.h"
#include "include/poisson.h"

const char *program_name = "cudapaste 1.0";
const char *program_bug_address = "killavus@gmail.com";
static char doc[] = "Cudapaste -- paste images into each other with CUDA acceleration";
static char args_doc[] = "BGIMAGE PASTEIMAGE MASKIMAGE XPOS YPOS";

static struct argp_option options[] = {
  { "type", 't', "TYPE", 0, "Paste method to be used. Possible values: naive, poisson" },
  { "iterations", 'i', "NITER", 0, "How many iterations there should be for a given algorithm. For naive this option is ignored." },
  { "outfile", 'o', "OUTFILE", 0, "Name of the file where output should be saved." },
  { 0 }
};

struct arguments {
  char *type;
  char *outfile;
  size_t iterations;
  size_t xpos;
  size_t ypos;
  char *bgimage;
  char *pasteimage;
  char *maskimage;
};

static error_t parse_opt (int key, char *arg, struct argp_state *state) {
  struct arguments *arguments = (struct arguments*) state->input;

  switch (key) {
    case 't':
      arguments->type = arg;    
      break;
    case 'o':
      arguments->outfile = arg;
      break;
    case 'i':
      arguments->iterations = atoi(arg);
      break;
    case ARGP_KEY_ARG:
      switch (state->arg_num) {
        case 0:
          arguments->bgimage = arg;
          break;
        case 1:
          arguments->pasteimage = arg;
          break;
        case 2:
          arguments->maskimage = arg;
          break;
        case 3:
          arguments->xpos = atoi(arg);
          break;
        case 4:
          arguments->ypos = atoi(arg);
          break;
        default:
          fprintf(stderr, "Ignoring excessive positional parameter: %s", arg);
          break;
      }
      break;
    case ARGP_KEY_END:
      if (state->arg_num < 4) {
        argp_usage(state);        
      }
      break;
    default:
      return ARGP_ERR_UNKNOWN;
  }

  return 0;
}

static struct argp argp = { options, parse_opt, args_doc, doc };

int main(int argc, char **argv) {
  struct arguments arguments;
  arguments.outfile = (char*) "out.png";
  arguments.iterations = 200;
  arguments.type = (char*) "naive";

  argp_parse(&argp, argc, argv, 0, 0, &arguments);
  
  HostImage *paste_raw = load_host_image(arguments.pasteimage);
  HostImage *mask_raw = load_host_image(arguments.maskimage);
  HostImage *bg = load_host_image(arguments.bgimage);
  HostImage *paste = resize_host_image(
    paste_raw,
    bg->w,
    bg->h,
    arguments.xpos,
    arguments.ypos
  );

  HostImage *mask = resize_host_image(
    mask_raw,
    bg->w,
    bg->h,
    arguments.xpos,
    arguments.ypos
  );

  drop_host_image(paste_raw);
  drop_host_image(mask_raw);

  HostImage *result = NULL;

  if (strcmp(arguments.type, "naive") == 0) {
    printf("Using naive method...\n"); 
    result = run_naive_method(bg, paste, mask);
  }
  
  if (strcmp(arguments.type, "poisson") == 0) {
    printf("Using poisson gradient descent method...\n");
    result = run_poisson(bg, paste, mask, arguments.iterations);
  }

  printf("Saved result in %s...\n", arguments.outfile);
  save_host_image(result, arguments.outfile);

  return 0;
}
