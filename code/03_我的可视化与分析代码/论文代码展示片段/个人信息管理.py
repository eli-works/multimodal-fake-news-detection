

@router.patch("/me", response_model=APIResponse[ProfileData])
async def update_profile(payload: UpdateProfileRequest, db: AsyncSession = Depends(get_db), user: User = Depends(get_current_user)):
    fields = payload.model_fields_set
    for k in ("email", "gender", "birthday", "phone", "bio"):
        if k in fields:
            setattr(user, k, getattr(payload, k))

    await db.commit()
    await db.refresh(user)
    return APIResponse(data=ProfileData.model_validate(user))

