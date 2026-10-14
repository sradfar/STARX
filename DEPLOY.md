# Publish STARX at no cost

## Files are ready

The deployable app is the `starx_streamlit` directory. The accompanying
`STARX_Streamlit_deployment.zip` contains only deployable source, model assets,
public scientific evidence, tests and documentation. It excludes virtual
environments, unpublished manuscripts, secrets, local runtime records and test
outputs. You do not need to regenerate or retrain the packaged paper models.

## 1. Create the repository

Signed into GitHub as **sradfar**, open https://github.com/new.

- Owner: **sradfar**
- Repository: **STARX** (if available; do not replace an existing project)
- Description: **Coastal forecast decision support with short-record model adaptation**
- Visibility: **Public**
- Leave repository initialization empty when uploading the prepared local Git repository.

The app directory can then be committed and pushed to the new repository. Use
GitHub Desktop's **Add existing repository → Publish repository** if you prefer
a graphical workflow. The standalone deployment ZIP can also be extracted to a
new folder and added to GitHub Desktop. Preserve the `assets`, `.streamlit`,
`starx`, and `tests` folders. Never upload `.venv`, `.runtime`, credentials or
the parent research workspace.

## 2. Deploy from the connected Streamlit account

At https://share.streamlit.io:

1. Select **Create app** and deploy an existing GitHub repository.
2. Repository: **sradfar/STARX**.
3. Branch: **main**.
4. Main file: **app.py**.
5. Advanced settings → Python: **3.12**. No secrets are needed.
6. Choose an available free subdomain, for example **starx-coastal**.
7. Select **Deploy** and wait for the dependencies and model assets to load.

The domain is allocated only when Streamlit accepts the chosen name. No custom
domain purchase or paid compute selection is required.

## 3. Verify the public tool

- Fetch current NOAA/GFS inputs at one station and execute all five leads.
- Check the UTC origin/valid times, surge thresholds and total-water-level datum.
- Open transfer diagnostics, historical SHAP profiles and paper evidence.
- At **Update models**, select **Boston**, use the included real target-record
  example, and run candidate selection. Allow several minutes on free CPUs.
- Download and restore the resulting model ZIP.
- Download a forecast and decision brief; save a monitoring checkpoint.
- Open the public URL from a signed-out browser.

## What persists

The packaged models, evidence and source persist in GitHub. Visitor-created
models and monitoring history persist through downloaded model/checkpoint ZIPs.
Community Cloud can sleep or restart, so this deployment does not promise
unattended collection while inactive. No API keys or external database accounts
are required for the included functionality.
