



def get_model_config(args):

	if args.backbone == "resNet2":
		args.patch_size = 13
		args.randomCrop = 11
		args.pca = True
		args.components = 15
		args.lambda_orth = 0.1
		args.lambda_kl = 0
		args.step_size = 30
		args.gamma = 0.7

	elif args.backbone == "vit_dino_s":
		args.patch_size = 6
		args.randomCrop = 4
		args.pca = True
		args.components = 15
		args.lambda_orth = 0.1
		args.lambda_kl = 0
		args.step_size = 30
		args.gamma = 0.7



	else:
		raise ValueError(f"Unsupported backbone: {args.backbone}")