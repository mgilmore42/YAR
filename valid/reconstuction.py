
def reconstruction(model, logger, dataloader, optimizer, transforms):

	running_loss = 0.0

	model.train()

	for i, (img, _) in enumerate(test_loader):

		optimizer.zero_grad(set_to_none=True)

		img = transforms(img.cuda().float()).bfloat16() * loss_mask

		out = output_normalize(model(img)) * loss_mask

		loss = criterion(out[:,:,loss_indecies], img[:,:,loss_indecies])

		if torch.isnan(loss):
			print(f'batch {i} NaN loss')
		else:
			loss.backward()

		optimizer.step()

		running_loss += img.shape[0] * loss.item()

		del loss, out, img

	epoch_loss = running_loss/len(dataset)

	model.eval()